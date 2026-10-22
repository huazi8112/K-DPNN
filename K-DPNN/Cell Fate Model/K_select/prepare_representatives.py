"""从原始四列训练参数中按maximin选择R4使用的30组代表条件。

仅生成代表参数CSV；不执行SDE模拟或GMM拟合。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from density_utils import stable_distribution


def representative_indices(parameters: np.ndarray, count: int) -> list[int]:
    """在标准化参数空间中执行确定性的 farthest-point/maximin 选点。"""
    count = min(count, len(parameters))
    center = parameters.mean(axis=0)
    scale = parameters.std(axis=0, ddof=1)
    scale[scale <= 1e-12] = 1.0
    standardized = (parameters - center) / scale

    selected = [int(np.argmin(np.sum(standardized * standardized, axis=1)))]
    min_distance = np.sum(
        (standardized - standardized[selected[0]]) ** 2,
        axis=1,
    )
    min_distance[selected[0]] = -np.inf

    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.sum(
            (standardized - standardized[next_index]) ** 2,
            axis=1,
        )
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True,
                        help="原始ps_train.csv路径，四列参数的顺序为a、b、降解率、噪声强度")
    parser.add_argument("--stable-states", type=Path, required=True,
                        help="与参数表行顺序对应的train_all_xstable_and_w.mat路径")
    parser.add_argument("--output", type=Path, required=True,
                        help="代表参数CSV的保存路径")
    args = parser.parse_args()
    parameters = np.genfromtxt(args.parameters, delimiter=",", skip_header=1)
    if (parameters.ndim != 2 or parameters.shape[1] != 4
            or len(parameters) < 30 or not np.isfinite(parameters).all()):
        parser.error("参数表必须含表头、至少30行以及四列有限数值")
    mat = loadmat(args.stable_states, squeeze_me=True)
    states_all, weights_all = mat["all_xstable"], mat["all_w"]
    if len(states_all) != len(parameters) or len(weights_all) != len(parameters):
        parser.error("稳定点MAT中的all_xstable、all_w必须与参数表逐行对应")
    rows = []
    for order, index in enumerate(representative_indices(parameters, 30), start=1):
        states, _ = stable_distribution(states_all[index], weights_all[index])
        rows.append({
            "selection_order": order,
            "condition_id": f"train_{index + 1}",
            "source_row_1_based": index + 1,
            **{f"parameter{i+1}": float(value) for i, value in enumerate(parameters[index])},
            "n_stable_states": len(states),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存30组代表参数：{args.output.resolve()}")


if __name__ == "__main__":
    main()
