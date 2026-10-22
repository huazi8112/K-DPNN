"""ABC各100万点的右闭稳态窗口实验，并按90%覆盖率选择K。

generate: 每个参数、每个A/B/C组各模拟5000条独立轨迹，在(0.8T,T]按0.1取200点。
evaluate: 用A组全部100万点拟合K=3,4,5，分别在B、C组计算Hellinger距离。
evaluate同时汇总：以30个固定代表参数为覆盖单位，选择至少27组满足平均H<=0.10的最小K。

数据生成需要含numba的环境；拟合需要含scikit-learn的环境。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

import density_utils as helpers
from density_utils import stable_distribution


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results"
CANDIDATE_K = (3, 4, 5)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def read_representatives(source: Path) -> tuple[list[dict], Path]:
    source = source.resolve()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or [int(row["selection_order"]) for row in rows] != list(range(1, 31)):
        raise ValueError("代表参数表不是预期的30组顺序")
    return rows, source



def kernel_factory():
    from numba import njit, prange

    @njit(cache=True, parallel=True)
    def kernel(
        trajectory_offset,
        n_cells,
        n_steps,
        steady_step,
        sample_step,
        snapshots,
        delta,
        a,
        b,
        degradation,
        noise,
        half,
        hill,
        states,
        cdf,
        initial_sigma,
        seed,
    ):
        output = np.empty((n_cells, snapshots, 2), dtype=np.float32)
        half_power = half**hill
        sqrt_noise = math.sqrt(delta) * noise
        milstein = 0.5 * delta * noise * noise
        for local_cell in prange(n_cells):
            global_cell = trajectory_offset + local_cell
            np.random.seed((seed + global_cell * 104729) % 2147483647)
            draw = np.random.random()
            state_index = 0
            while state_index < cdf.size - 1 and draw > cdf[state_index]:
                state_index += 1
            x1, x2 = -1.0, -1.0
            while x1 <= 0 or x2 <= 0:
                x1 = states[state_index, 0] + initial_sigma * np.random.randn()
                x2 = states[state_index, 1] + initial_sigma * np.random.randn()
            stored = 0
            for step in range(1, n_steps + 1):
                power1, power2 = x1**hill, x2**hill
                f1 = a * power1 / (half_power + power1) + b * half_power / (half_power + power2) - degradation * x1
                f2 = a * power2 / (half_power + power2) + b * half_power / (half_power + power1) - degradation * x2
                r0, r1 = np.random.randn(), np.random.randn()
                r2, r3 = np.random.randn(), np.random.randn()
                # 保持项目既有的SDE积分更新不变。
                x1 = abs(x1 + delta * f1 + sqrt_noise * r0 + milstein * (r1 * r1 - 1.0))
                x2 = abs(x2 + delta * f2 + sqrt_noise * r2 + milstein * (r3 * r3 - 1.0))
                # 严格使用(0.8T,T]：不保存80.0，保存80.1至100.0。
                if step > steady_step and (step - steady_step) % sample_step == 0:
                    output[local_cell, stored, 0] = x1
                    output[local_cell, stored, 1] = x2
                    stored += 1
        return output

    return kernel


def simulation_config(source: Path, threads: int, actual_threads: int) -> dict:
    return {
        "n_parameters": 30,
        "groups_per_condition": ["A", "B", "C"],
        "trajectories_per_group": 5000,
        "snapshots_per_trajectory": 200,
        "points_per_group": 1000000,
        "point_shape_per_group": [1000000, 2],
        "T": 100.0,
        "integration_dt": 0.01,
        "steady_window": "(0.8T,T]",
        "steady_fraction": 0.8,
        "sample_interval": 0.1,
        "sample_times_first_last": [80.1, 100.0],
        "sample_time_count": 200,
        "dtype": "float32",
        "storage_order": "trajectory-major; 200 consecutive rows belong to one trajectory",
        "base_seed": 2026100201,
        "seed_formula": "(base_seed + 1000003*source_row + global_trajectory_id*104729) % 2147483647",
        "group_global_trajectory_ids": "A:0..4999, B:5000..9999, C:10000..14999",
        "ABC_use_disjoint_trajectories": True,
        "time_points_within_one_trajectory_are_correlated": True,
        "state_initial_sigma": 0.1,
        "half": 0.5,
        "hill": 4,
        "grid": [0.0, 6.0, 0.05],
        "density_epsilon": 1e-9,
        "smoothing_bins": 0,
        "reference_parameters_sha256": sha256(source),
        "requested_numba_threads": threads,
        "actual_numba_threads": actual_threads,
    }


def generate(args, rows: list[dict], source: Path) -> None:
    from numba import get_num_threads, set_num_threads
    from scipy.io import loadmat

    start = time.perf_counter()
    set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = simulation_config(source, args.threads, get_num_threads())
    config_path = output / "simulation_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("现有模拟配置不同，请使用新的输出目录")
    atomic_json(config_path, config)
    (output / "representative_parameters.csv").write_bytes(source.read_bytes())
    sample_times = np.arange(8010, 10001, 10, dtype=np.int64) * 0.01
    if len(sample_times) != 200 or not np.isclose(sample_times[0], 80.1) or not np.isclose(sample_times[-1], 100.0):
        raise AssertionError("采样时刻构造错误")
    np.save(output / "sample_times.npy", sample_times)

    mat = loadmat(args.stable_states, squeeze_me=True)
    kernel = kernel_factory()
    stable_test = np.array([[1.125, 1.125]], dtype=np.float64)
    cdf_test = np.array([1.0], dtype=np.float64)
    check_start = time.perf_counter()
    deterministic = kernel(0, 2, 100, 80, 10, 2, 0.01, 0.9, 0.9, 0.8, 0.0, 0.5, 4, stable_test, cdf_test, 0.0, 1234567)
    if deterministic.shape != (2, 2, 2) or not np.allclose(deterministic, 1.125):
        raise AssertionError("确定性积分或右闭窗口检查失败")
    together = kernel(0, 2, 100, 80, 10, 2, 0.01, 0.9, 0.9, 0.8, 0.25, 0.5, 4, stable_test, cdf_test, 0.1, 1234567)
    separate = np.concatenate([
        kernel(i, 1, 100, 80, 10, 2, 0.01, 0.9, 0.9, 0.8, 0.25, 0.5, 4, stable_test, cdf_test, 0.1, 1234567)
        for i in range(2)
    ])
    if not np.array_equal(together, separate):
        raise AssertionError("分块模拟不能复现同一轨迹")
    compile_check_seconds = time.perf_counter() - check_start

    completed_conditions = 0
    for row in rows:
        condition = row["condition_id"]
        row_id = int(row["source_row_1_based"])
        folder = output / condition
        folder.mkdir(exist_ok=True)
        marker = folder / "generation_complete.json"
        if marker.exists():
            completed_conditions += 1
            print(f"[{completed_conditions}/30] {condition}: 三组数据已存在，跳过生成", flush=True)
            continue
        stable, cdf = stable_distribution(mat["all_xstable"][row_id - 1], mat["all_w"][row_id - 1])
        parameter = [float(row[f"parameter{i}"]) for i in range(1, 5)]
        condition_start = time.perf_counter()
        timing_rows = []
        for group_index, group in enumerate(("A", "B", "C")):
            data_path = folder / f"{condition}_{group}_n1000000_window_right_closed.npy"
            progress_path = folder / f"{group}_generation_progress.json"
            progress = (
                json.loads(progress_path.read_text(encoding="utf-8"))
                if progress_path.exists()
                else {"completed_trajectories": 0, "compute_seconds": 0.0, "write_seconds": 0.0}
            )
            completed = int(progress["completed_trajectories"])
            saved = np.lib.format.open_memmap(
                data_path,
                mode="r+" if completed else "w+",
                dtype=np.float32,
                shape=(1000000, 2),
            )
            seed = config["base_seed"] + 1000003 * row_id
            while completed < 5000:
                n_chunk = min(args.chunk_trajectories, 5000 - completed)
                compute_start = time.perf_counter()
                chunk = kernel(
                    group_index * 5000 + completed,
                    n_chunk,
                    10000,
                    8000,
                    10,
                    200,
                    0.01,
                    *parameter,
                    0.5,
                    4,
                    stable,
                    cdf,
                    0.1,
                    seed,
                )
                compute_seconds = time.perf_counter() - compute_start
                if not np.isfinite(chunk).all() or (chunk < 0).any():
                    raise ValueError(f"{condition} {group} 出现无效模拟值")
                write_start = time.perf_counter()
                saved[completed * 200 : (completed + n_chunk) * 200] = chunk.reshape(-1, 2)
                saved.flush()
                write_seconds = time.perf_counter() - write_start
                completed += n_chunk
                progress.update(
                    completed_trajectories=completed,
                    compute_seconds=float(progress["compute_seconds"] + compute_seconds),
                    write_seconds=float(progress["write_seconds"] + write_seconds),
                    file=data_path.name,
                    complete=completed == 5000,
                )
                atomic_json(progress_path, progress)
            del saved
            timing_rows.append(
                {
                    "group": group,
                    **progress,
                    "bytes": data_path.stat().st_size,
                    "sha256": sha256(data_path),
                }
            )
        generation = {
            "condition_id": condition,
            "selection_order": int(row["selection_order"]),
            "source_row_1_based": row_id,
            "parameter": parameter,
            "groups": timing_rows,
            "total_pure_simulation_seconds": float(sum(item["compute_seconds"] for item in timing_rows)),
            "total_write_seconds": float(sum(item["write_seconds"] for item in timing_rows)),
            "condition_generation_wall_seconds": time.perf_counter() - condition_start,
            "compile_and_integrator_check_seconds": compile_check_seconds,
            "python_executable": sys.executable,
            "complete": True,
        }
        atomic_json(marker, generation)
        completed_conditions += 1
        atomic_json(
            output / "generation_progress.json",
            {
                "completed_conditions": completed_conditions,
                "total_conditions": 30,
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
        print(
            f"[{completed_conditions}/30] {condition}: A/B/C各1,000,000点生成完成，"
            f"纯模拟{generation['total_pure_simulation_seconds']:.3f}s",
            flush=True,
        )
    generation_markers = [
        json.loads((output / row["condition_id"] / "generation_complete.json").read_text(encoding="utf-8"))
        for row in rows
    ]
    atomic_json(
        output / "generation_summary.json",
        {
            "complete": True,
            "n_conditions": 30,
            "n_files": 90,
            "n_points_per_file": 1000000,
            "total_points": 90000000,
            "latest_invocation_wall_seconds": time.perf_counter() - start,
            "sum_condition_generation_wall_seconds": float(
                sum(item["condition_generation_wall_seconds"] for item in generation_markers)
            ),
            "sum_pure_simulation_seconds": float(
                sum(item["total_pure_simulation_seconds"] for item in generation_markers)
            ),
            "compile_and_integrator_check_seconds_latest_invocation": compile_check_seconds,
        },
    )


def evaluate_condition(task: dict) -> dict:
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    output = Path(task["output_dir"])
    condition = task["condition_id"]
    row_id = int(task["source_row_1_based"])
    folder = output / condition
    marker_path = folder / "evaluation_complete.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return {"condition_id": condition, "wall_seconds": marker["evaluation_wall_seconds"], "resumed": True}

    config = json.loads((output / "simulation_config.json").read_text(encoding="utf-8"))
    generation = json.loads((folder / "generation_complete.json").read_text(encoding="utf-8"))
    fit_config = {
        "fit_data": "all A right-closed steady-window points; no subsampling",
        "candidate_K": list(CANDIDATE_K),
        "n_A": 1000000,
        "n_B": 1000000,
        "n_C": 1000000,
        "n_init": task["n_init"],
        "max_iter": 300,
        "reg_covar": 1e-6,
        "covariance_type": "full",
        "base_fit_seed": 41867031,
        "blas_threads_per_process": 1,
        "source_A_sha256": generation["groups"][0]["sha256"],
        "H_threshold": 0.10,
    }
    fit_config_path = folder / "fit_config.json"
    if fit_config_path.exists() and json.loads(fit_config_path.read_text(encoding="utf-8")) != fit_config:
        raise ValueError(f"{condition} 现有拟合配置不同")
    atomic_json(fit_config_path, fit_config)

    condition_start = time.perf_counter()
    edges = np.arange(0.0, 6.0 + 0.025, 0.05)
    densities = {}
    coverages = {}
    data_paths = {}
    reference_start = time.perf_counter()
    for group in ("A", "B", "C"):
        data_path = folder / f"{condition}_{group}_n1000000_window_right_closed.npy"
        points = np.load(data_path, mmap_mode="r")
        if points.shape != (1000000, 2) or points.dtype != np.float32:
            raise ValueError(f"{condition} {group} 数据形状或类型错误")
        density, coverage = helpers.empirical_density(points, edges, config["density_epsilon"])
        if coverage < 0.999:
            raise ValueError(f"{condition} {group} 网格覆盖不足：{coverage}")
        densities[group] = density
        coverages[group] = coverage
        data_paths[group] = data_path
        del points
    reference_pairs = [
        {"pair": first + second, "H": helpers.hellinger(densities[first], densities[second])}
        for first, second in (("A", "B"), ("A", "C"), ("B", "C"))
    ]
    write_csv(folder / "ABC_reference_hellinger.csv", reference_pairs)
    np.savez_compressed(folder / "ABC_probability_matrices.npz", edges=edges, **densities)
    reference_seconds = time.perf_counter() - reference_start

    load_start = time.perf_counter()
    a_points = np.array(np.load(data_paths["A"], mmap_mode="r"), dtype=np.float64, copy=True)
    a_load_seconds = time.perf_counter() - load_start
    rows_fit = []
    for k in CANDIDATE_K:
        result_path = folder / f"gmm_K{k}_result.json"
        if result_path.exists():
            rows_fit.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        fit_seed = fit_config["base_fit_seed"] + row_id * 1009 + k * 7919
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=task["n_init"],
            max_iter=300,
            reg_covar=1e-6,
            random_state=fit_seed,
        )
        fit_start = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(a_points)
        fit_seconds = time.perf_counter() - fit_start
        if not model.converged_ or (np.linalg.eigvalsh(model.covariances_) <= 0).any():
            raise ValueError(f"{condition} K={k} GMM未收敛或协方差无效")
        integration_start = time.perf_counter()
        raw_mass, model_coverage = helpers.gmm_grid_mass(
            model.weights_, model.means_, model.covariances_, edges
        )
        model_density = helpers.normalized_density(raw_mass, config["density_epsilon"])
        h_b = helpers.hellinger(model_density, densities["B"])
        h_c = helpers.hellinger(model_density, densities["C"])
        np.savez_compressed(
            folder / f"gmm_K{k}_parameters.npz",
            weights=model.weights_,
            means=model.means_,
            covariances=model.covariances_,
            grid_density=model_density,
            grid_edges=edges,
            grid_mass_before_renormalization=raw_mass,
        )
        result = {
            "condition_id": condition,
            "selection_order": int(task["selection_order"]),
            "source_row_1_based": row_id,
            "K": k,
            "n_A": 1000000,
            "n_B": 1000000,
            "n_C": 1000000,
            "H_GMM_B": h_b,
            "H_GMM_C": h_c,
            "H_GMM_mean": (h_b + h_c) / 2.0,
            "pair_affinity_mean": 1.0 - (h_b * h_b + h_c * h_c) / 2.0,
            "H_reference_AB": reference_pairs[0]["H"],
            "H_reference_AC": reference_pairs[1]["H"],
            "H_reference_BC": reference_pairs[2]["H"],
            "converged": bool(model.converged_),
            "n_iter_best_initialization": int(model.n_iter_),
            "fit_seed": fit_seed,
            "fit_seconds": fit_seconds,
            "integration_and_save_seconds": time.perf_counter() - integration_start,
            "empirical_grid_coverage_A": coverages["A"],
            "empirical_grid_coverage_B": coverages["B"],
            "empirical_grid_coverage_C": coverages["C"],
            "gmm_grid_coverage_before_renormalization": model_coverage,
        }
        atomic_json(result_path, result)
        rows_fit.append(result)
        write_csv(folder / "gmm_K3_K5_results.csv", sorted(rows_fit, key=lambda item: item["K"]))
    del a_points
    rows_fit.sort(key=lambda item: item["K"])
    passing = [item["K"] for item in rows_fit if item["H_GMM_mean"] <= 0.10]
    summary = {
        "condition_id": condition,
        "selection_order": int(task["selection_order"]),
        "parameter": [float(task[f"parameter{i}"]) for i in range(1, 5)],
        "n_points_per_group": 1000000,
        "n_trajectories_per_group": 5000,
        "n_snapshots_per_trajectory": 200,
        "sample_window": "(0.8T,T]",
        "reference_pairs": reference_pairs,
        "reference_matrix_compute_seconds": reference_seconds,
        "A_load_seconds": a_load_seconds,
        "grid_coverages": coverages,
        "GMM_fit_seconds_total": float(sum(item["fit_seconds"] for item in rows_fit)),
        "GMM_results": rows_fit,
        "smallest_passing_K_for_this_condition_only": min(passing) if passing else None,
        "evaluation_wall_seconds": time.perf_counter() - condition_start,
        "python_executable": sys.executable,
        "complete": True,
    }
    atomic_json(marker_path, summary)
    return {"condition_id": condition, "wall_seconds": summary["evaluation_wall_seconds"], "resumed": False}


def choose_k(matrix: np.ndarray, ks: list[int], threshold: float = 0.10, coverage: float = 0.90):
    required = math.ceil(coverage * len(matrix) - 1e-12)
    for column, k in enumerate(ks):
        if int((matrix[:, column] <= threshold).sum()) >= required:
            return int(k)
    return None


def build_summary(details: list[dict], bootstrap: int, seed: int) -> tuple[list[dict], dict, list[dict]]:
    by_condition = {}
    for row in details:
        by_condition.setdefault(row["condition_id"], {})[int(row["K"])] = row
    if len(by_condition) != 30 or any(set(values) != set(CANDIDATE_K) for values in by_condition.values()):
        raise ValueError("30组参数×3个K的结果矩阵不完整")
    ordered_conditions = sorted(by_condition, key=lambda key: by_condition[key][3]["selection_order"])
    matrix = np.array(
        [[by_condition[condition][k]["H_GMM_mean"] for k in CANDIDATE_K] for condition in ordered_conditions],
        dtype=np.float64,
    )
    rank_index = math.ceil(0.90 * len(matrix) - 1e-12) - 1
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matrix), size=(bootstrap, len(matrix)))
    boot = matrix[indices]
    boot_counts = (boot <= 0.10).sum(axis=1)
    boot_rank = np.sort(boot, axis=1)[:, rank_index, :]
    summary_rows = []
    for column, k in enumerate(CANDIDATE_K):
        values = matrix[:, column]
        count = int((values <= 0.10).sum())
        summary_rows.append(
            {
                "K": k,
                "n_conditions": len(values),
                "mean_H": float(values.mean()),
                "median_H": float(np.median(values)),
                "q90_H_linear": float(np.quantile(values, 0.90)),
                "coverage_rank_H_27th": float(np.sort(values)[rank_index]),
                "max_H": float(values.max()),
                "n_pass_H_le_0_10": count,
                "pass_fraction": count / len(values),
                "bootstrap_rank_H_lower95": float(np.quantile(boot_rank[:, column], 0.025)),
                "bootstrap_rank_H_upper95": float(np.quantile(boot_rank[:, column], 0.975)),
                "bootstrap_probability_meeting_90pct_coverage": float(np.mean(boot_counts[:, column] >= 27)),
            }
        )
    frequencies = {str(k): 0 for k in CANDIDATE_K}
    frequencies["no_feasible_K_at_most_5"] = 0
    for sample in boot:
        selected = choose_k(sample, list(CANDIDATE_K))
        frequencies[str(selected) if selected is not None else "no_feasible_K_at_most_5"] += 1
    selection = {
        "selected_K": choose_k(matrix, list(CANDIDATE_K)),
        "n_conditions": 30,
        "threshold_H": 0.10,
        "coverage_required": 0.90,
        "required_pass_count": 27,
        "coverage_value_per_condition": "mean(H_GMM_B,H_GMM_C)",
        "candidate_K": list(CANDIDATE_K),
        "selection_formula": "min K in {3,4,5} with count_i[mean(H_GMM_B,H_GMM_C)<=0.10] >= 27",
        "no_feasible_policy": "report insufficient capacity; do not force K=5",
        "bootstrap_repeats": bootstrap,
        "bootstrap_selection_frequency": frequencies,
        "bootstrap_selection_fraction": {key: value / bootstrap for key, value in frequencies.items()},
        "bootstrap_interpretation": "fixed-design descriptive sensitivity, not a random-population guarantee",
    }
    sensitivity_rows = []
    for threshold in (0.05, 0.06, 0.07, 0.08, 0.09, 0.095, 0.10, 0.105, 0.11, 0.12, 0.13, 0.15):
        for coverage in (0.80, 0.90, 0.95, 1.00):
            sensitivity_rows.append(
                {
                    "H_threshold": threshold,
                    "minimum_affinity": 1.0 - threshold * threshold,
                    "required_coverage": coverage,
                    "required_pass_count": math.ceil(coverage * 30 - 1e-12),
                    "selected_K_at_most_5": choose_k(matrix, list(CANDIDATE_K), threshold, coverage),
                }
            )
    return summary_rows, selection, sensitivity_rows


def finalize(args, rows: list[dict], source: Path, evaluation_wall_seconds: float) -> None:
    output = args.output_dir.resolve()
    details = []
    condition_summaries = []
    for row in rows:
        folder = output / row["condition_id"]
        marker = json.loads((folder / "evaluation_complete.json").read_text(encoding="utf-8"))
        condition_summaries.append(marker)
        for k in CANDIDATE_K:
            details.append(json.loads((folder / f"gmm_K{k}_result.json").read_text(encoding="utf-8")))
    details.sort(key=lambda item: (item["selection_order"], item["K"]))
    if len(details) != 90 or any(not row["converged"] for row in details):
        raise ValueError("GMM结果数量或收敛状态异常")
    write_csv(output / "abc_million_gmm_details.csv", details)
    summary_rows, selection, sensitivity_rows = build_summary(details, args.bootstrap, 2026100301)
    write_csv(output / "coverage_summary_by_K.csv", summary_rows)
    write_csv(output / "threshold_coverage_sensitivity.csv", sensitivity_rows)

    condition_rows = []
    for marker in sorted(condition_summaries, key=lambda item: item["selection_order"]):
        row = {
            "selection_order": marker["selection_order"],
            "condition_id": marker["condition_id"],
            "parameter1": marker["parameter"][0],
            "parameter2": marker["parameter"][1],
            "parameter3": marker["parameter"][2],
            "parameter4": marker["parameter"][3],
            "H_reference_AB": marker["reference_pairs"][0]["H"],
            "H_reference_AC": marker["reference_pairs"][1]["H"],
            "H_reference_BC": marker["reference_pairs"][2]["H"],
            "smallest_passing_K_condition_only": marker["smallest_passing_K_for_this_condition_only"],
            "GMM_fit_seconds_total": marker["GMM_fit_seconds_total"],
            "evaluation_wall_seconds": marker["evaluation_wall_seconds"],
        }
        for result in marker["GMM_results"]:
            row[f"K{result['K']}_H_GMM_mean"] = result["H_GMM_mean"]
        condition_rows.append(row)
    write_csv(output / "condition_summary.csv", condition_rows)

    generation_summary = json.loads((output / "generation_summary.json").read_text(encoding="utf-8"))
    selection.update(
        {
            "all_fits_converged": True,
            "new_independent_ABC_SDE_data": True,
            "sample_window": "(0.8T,T]",
            "points_per_group": 1000000,
            "trajectories_per_group": 5000,
            "snapshots_per_trajectory": 200,
            "generation_sum_condition_wall_seconds": generation_summary["sum_condition_generation_wall_seconds"],
            "generation_recorded_pure_simulation_seconds": generation_summary["sum_pure_simulation_seconds"],
            "evaluation_stage_wall_seconds": evaluation_wall_seconds,
            "sum_condition_evaluation_wall_seconds": float(sum(item["evaluation_wall_seconds"] for item in condition_summaries)),
            "sum_GMM_fit_seconds": float(sum(item["fit_seconds"] for item in details)),
            "python_executable": sys.executable,
            "reference_parameters_sha256": sha256(source),
        }
    )
    atomic_json(output / "budget_selection.json", selection)

    integrity = {
        "complete": True,
        "n_conditions": len(condition_summaries),
        "n_raw_files": 90,
        "n_raw_points_total": 90000000,
        "n_gmm_fits": len(details),
        "all_fits_converged": all(item["converged"] for item in details),
        "all_empirical_grid_coverages_at_least_0_999": all(
            min(item["empirical_grid_coverage_A"], item["empirical_grid_coverage_B"], item["empirical_grid_coverage_C"]) >= 0.999
            for item in details
        ),
        "all_condition_K_complete": all(
            [item["K"] for item in marker["GMM_results"]] == list(CANDIDATE_K)
            for marker in condition_summaries
        ),
        "sample_times_exact": bool(
            np.array_equal(np.load(output / "sample_times.npy"), np.arange(8010, 10001, 10) * 0.01)
        ),
        "selection_recomputed": selection["selected_K"],
    }
    atomic_json(output / "final_integrity_checks.json", integrity)
    if not all(value is True for key, value in integrity.items() if key.startswith("all_") or key == "sample_times_exact"):
        raise ValueError("最终完整性检查失败")

    lines = [
        "# ABC各百万点右闭稳态窗口：90%覆盖率K选择",
        "",
        "## 实验定义",
        "",
        "30组固定maximin代表参数；每组参数重新生成A、B、C三组互不重复的轨迹。",
        "每组5000条轨迹，在(0.8T,T]内每隔0.1取点，即80.1至100.0共200个时刻，恰好1,000,000个二维状态。",
        "同一轨迹内的200个时点有相关性，因此100万是状态点数，独立轨迹数为5000。",
        "A组全部100万点用于完整协方差GMM拟合；B、C各100万点仅用于独立评价。K=3、4、5，10次初始化，最多300次EM。",
        "每个参数的覆盖值为B、C两个Hellinger距离的平均；H≤0.10视为通过。30组中至少27组通过时达到90%覆盖率。",
        "",
        "## 结果",
        "",
        "| K | 平均H | 中位H | 第27小H | 通过组数 | 覆盖率 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['K']} | {row['mean_H']:.6f} | {row['median_H']:.6f} | "
            f"{row['coverage_rank_H_27th']:.6f} | {row['n_pass_H_le_0_10']}/30 | {row['pass_fraction']:.1%} |"
        )
    selected_text = str(selection["selected_K"]) if selection["selected_K"] is not None else "K≤5无可行值"
    reference_values = np.array(
        [[row["H_reference_AB"], row["H_reference_AC"], row["H_reference_BC"]] for row in condition_rows]
    )
    lines += [
        "",
        f"按预先固定的规则，最优K结果为：**{selected_text}**。",
        "",
        "A/B/C经验参考分布两两Hellinger的30组均值分别为："
        f"AB={reference_values[:, 0].mean():.6f}，AC={reference_values[:, 1].mean():.6f}，BC={reference_values[:, 2].mean():.6f}。",
        "",
        "## 耗时与核查",
        "",
        f"30组数据生成的累计纯模拟计时（含断点前后已完成分块）：{generation_summary['sum_pure_simulation_seconds']:.3f}秒。",
        f"GMM评价阶段墙钟时间：{evaluation_wall_seconds:.3f}秒。",
        f"90次GMM拟合累计CPU墙钟时间（各任务相加）：{selection['sum_GMM_fit_seconds']:.3f}秒。",
        "90个GMM均收敛；A/B/C文件、采样时刻、结果矩阵和网格覆盖率均通过完整性检查。",
        "",
        "这里的90%是30个固定代表参数上的经验覆盖率；bootstrap只用于描述这30组固定设计的敏感性，不解释为总体概率保证。",
    ]
    (output / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args, rows: list[dict], source: Path) -> None:
    output = args.output_dir.resolve()
    if not (output / "generation_summary.json").exists():
        raise FileNotFoundError("请先完整运行generate阶段")
    start = time.perf_counter()
    tasks = [
        {
            **row,
            "output_dir": str(output),
            "n_init": args.n_init,
        }
        for row in rows
    ]
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_condition, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            atomic_json(
                output / "evaluation_progress.json",
                {
                    "completed_conditions": completed,
                    "total_conditions": 30,
                    "elapsed_seconds": time.perf_counter() - start,
                    "latest_condition": result["condition_id"],
                },
            )
            print(
                f"[{completed}/30] {result['condition_id']}: K=3/4/5完成，"
                f"单条件{result['wall_seconds']:.1f}s"
                + ("（断点复用）" if result["resumed"] else ""),
                flush=True,
            )
    evaluation_wall_seconds = time.perf_counter() - start
    finalize(args, rows, source, evaluation_wall_seconds)
    print((output / "budget_selection.json").read_text(encoding="utf-8"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="模拟数据及评价文件的输出目录（默认：代码目录下的results）")
    parser.add_argument("--representatives", type=Path,
                        help="prepare_representatives.py生成的30组代表参数CSV；evaluate默认读取输出目录中的副本")
    parser.add_argument("--stable-states", type=Path,
                        help="generate所需的原始train_all_xstable_and_w.mat路径")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--chunk-trajectories", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.threads < 1 or args.chunk_trajectories < 1 or args.workers < 1 or args.n_init < 1:
        parser.error("线程、分块、进程和初始化次数必须为正整数")
    if args.bootstrap < 1:
        parser.error("bootstrap次数必须为正整数")
    if args.stage == "generate":
        if args.representatives is None or args.stable_states is None:
            parser.error("generate需要--representatives和--stable-states两个输入路径")
        if not args.stable_states.is_file():
            parser.error(f"稳定点输入不存在：{args.stable_states}")
    source_path = args.representatives or (args.output_dir / "representative_parameters.csv")
    if not source_path.is_file():
        parser.error(f"代表参数CSV不存在：{source_path}")
    rows, source = read_representatives(source_path)
    if args.stage == "generate":
        generate(args, rows, source)
    else:
        evaluate(args, rows, source)


if __name__ == "__main__":
    main()
