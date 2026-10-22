# Data and checkpoints

Paths below are relative to the repository root. Parameter CSVs contain a header and one row per parameter condition. Keep the original row order when matching CSV rows to MAT filenames.

## Inputs included in this snapshot

| Location | Included inputs |
|---|---|
| `Cell Fate Model/data_generate/` | `ps_train.csv` (1,000 rows), `ps_valid.csv` (100), `ps_test.csv` (500), and the three corresponding `*_all_xstable_and_w.mat` files |
| `Coupled Toggle-Switch Model/train_model/` | `ps_train.csv` (1,000 rows), `ps_valid.csv` (200), `ps_test.csv` (200) |
| Each model's `train_model/` | One archived main checkpoint; filenames are listed below |

The Cell Fate stable-state MAT files contain `all_xstable` and `all_w`, aligned with the corresponding parameter CSV. The K-selection code reads these precomputed initial-state inputs; a stable-point solver is not supplied in this snapshot.

## Required density datasets

The following dataset directories are referenced by the notebooks but are **not included**. Obtain the matching archived data or generate a new, explicitly identified dataset before training or reference-based evaluation.

| Model's `train_model/` directory | Expected density-directory pattern |
|---|---|
| Cell Fate | `density_results_ps_<split>_n_cells=50000_edges=0.05/` |
| Coupled Toggle-Switch | `density_results_ps_<split>_n_cells=50000_python_edges=0.1/` |
| MESC plotting notebooks | `density_results_ps_<split>_edges=0.1（未转置）/` |

Replace `<split>` with `train`, `valid`, or `test`. Preserve the full-width parentheses in the MESC names when using the current defaults.

Within a dataset directory, one-based filenames include literal braces, for example `ps_train{1}.mat`, `ps_valid{1}.mat`, and `ps_test{1}.mat`. The training loaders expect the reference matrix under the `density` key. Preserve each dataset's grid, array-axis convention, and normalization when moving files.

The updated Cell Fate generator also saves `sample_times`, `sampling_window`, and `sample_count_per_cell`. The matrix named `density` is normalized histogram probability mass, so its elements sum to one; a continuous density would additionally account for grid-cell area.

## Connecting Cell Fate generation to training

1. Run the generator from `Cell Fate Model/data_generate/`. It creates `density_results_ps_<split>_n_cells=<n_cells>/` under the current working directory.
2. The generator currently uses 10 training trajectories and 50,000 validation/test trajectories. Choose and record the intended budget before generation. A directory generated with 10 trajectories must retain that provenance.
3. Make the generated directories available to `train_model` and set each notebook's `mat_*_dir`/test-density path to the actual location. The generator omits the `_edges=0.05` suffix used by the historical training defaults; this is a path difference to resolve explicitly.
4. Set training/evaluation CSV paths to the existing `../data_generate/ps_<split>.csv`, or copy these small tables into `train_model` while preserving row order.
5. Give regenerated datasets a separate identity from archived training labels. Their numerical simulation protocol has changed; the supplied checkpoints have not been retrained or revalidated against these regenerated labels during repository preparation.

## Additional MESC inputs

The MESC plotting notebooks require parameter tables such as `ps_test.csv` and the corresponding reference matrices. Those parameter tables are absent from this snapshot. `retrain_save.ipynb` additionally refers to historical `ps_*_filtered.csv` and density directories ending in `_filtered`; these are also absent. The filtering rule, aligned row indices, and the exact two-dimensional projection must accompany any future MESC dataset release.

The Coupled Toggle-Switch and MESC SDE/projection data-generation programs are not present in this snapshot. No public data-download URL or DOI has been supplied yet.

## Archived main checkpoints

| Model | Filename under `train_model/` |
|---|---|
| Cell Fate | `best_model_in4_K4_256-512-256_testloss0.051024.pth` |
| Coupled Toggle-Switch | `best_model_in5_K4_512-1024-512_testloss0.084797.pth` |
| MESC | `best_model_in4_K3_512-1024-512_testloss0.097046.pth` |

These filenames identify the supplied archives. Report Hellinger distance separately from any training loss containing moment penalties; do not infer metric definitions from a filename suffix alone.
