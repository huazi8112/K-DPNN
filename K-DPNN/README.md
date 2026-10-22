# K-DPNN

Neural prediction of stationary probability densities and gene regulatory landscapes using weighted two-dimensional Gaussian mixtures.

[中文说明](README.zh-CN.md) · [Data and checkpoints](DATA.md) · [Reproducibility notes](REPRODUCIBILITY.md)

## Included models

| Model | Parameter inputs | Mixture components | Hidden layers |
|---|---:|---:|---|
| Cell Fate Model | 4 | 4 | 256–512–256 |
| Coupled Toggle-Switch Model | 5 | 4 | 512–1024–512 |
| MESC Model | 4 | 3 | 512–1024–512 |

The first model predicts a distribution in the original two-dimensional state space. The four-gene and 12-dimensional MESC models predict distributions in their two-dimensional projection spaces.

This repository snapshot includes training/plotting notebooks, three archived model checkpoints, the Cell Fate data-generation notebooks, and the Cell Fate K-selection code. **The precomputed density datasets are not included.** Training and comparisons against reference distributions require the inputs described in [DATA.md](DATA.md).

## Repository structure

```text
K-DPNN/
├── Cell Fate Model/
│   ├── data_generate/       # Parameter tables, initial states, and SDE notebooks
│   ├── K_select/            # Independent A/B/C capacity-selection experiment
│   └── train_model/         # Training, checkpoint, and plotting notebooks
├── Coupled Toggle-Switch Model/
│   └── train_model/
├── MESC Model/
│   └── train_model/
├── DATA.md
├── REPRODUCIBILITY.md
└── requirements.txt
```

## Installation

Use Python 3.10 or later. From the repository root:

```text
python -m pip install -r requirements.txt
```

The dependency list covers imports used by the included source code; it is not an exact environment lock for the archived experiments. Model loading supports CPU execution. Training code selects the available device according to its notebook configuration.

## Running the code

### Cell Fate data generation

Start Jupyter from the data-generation directory:

```text
cd "Cell Fate Model/data_generate"
jupyter lab
```

Run the desired `Simulation_density_train.ipynb`, `Simulation_density_valid.ipynb`, or `Simulation_density_test.ipynb`. These notebooks read the matching parameter CSV and stable-state MAT in the same working directory.

The current generator integrates from 0 to 100 using a step of 0.01 and retains **80.1, 80.2, …, 100.0**, giving **200 states per trajectory**. Its Euler–Maruyama random increment is `sqrt(2 * D * delta) * xi`, followed by the existing nonnegative reflection. Results are written to subdirectories of the current working directory.

The training generator is currently configured for **10 trajectories per parameter**, while the validation and test generators use **50,000**. Check `n_cells` before launching a production run. The generated directory names differ from those expected by the training notebooks; see [DATA.md](DATA.md).

### Cell Fate K selection

The inputs needed by this component are already present in the adjacent `data_generate` directory. From the repository root:

```text
cd "Cell Fate Model/K_select"
python prepare_representatives.py --parameters "../data_generate/ps_train.csv" --stable-states "../data_generate/train_all_xstable_and_w.mat" --output "data/representative_parameters.csv"
python select_k.py generate --representatives "data/representative_parameters.csv" --stable-states "../data_generate/train_all_xstable_and_w.mat"
python select_k.py evaluate
python plot_k_selection.py
```

This experiment uses 30 maximin representative conditions, independent A/B/C samples, candidate K values of 3, 4, and 5, and selects the smallest K for which at least 27/30 conditions have mean Hellinger distance at most 0.10. Each A/B/C group contains one million states from 5,000 trajectories and 200 retained times per trajectory.

**K selection preserves the archived simulation update, which differs from the revised data generator.** Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before comparing or combining the two protocols.

### Training and plotting

Place the required datasets as described in [DATA.md](DATA.md), then start Jupyter inside the corresponding `train_model` directory. The main notebooks are:

| Model | Training notebook | Plotting notebooks |
|---|---|---|
| Cell Fate | `Train_load_model.ipynb` | `Plot_compare_save.ipynb`, `mean_variance_scatter.ipynb`, `marginal_distribution_comparison.ipynb` |
| Coupled Toggle-Switch | `Train_plot.ipynb` | `Plot_compare_save.ipynb`, `mean_variance_scatter.ipynb`, `marginal_distribution_comparison.ipynb` |
| MESC | `retrain_save.ipynb` | `Plot_compare_save.ipynb`, `mean_covariance_scatter.ipynb`, `marginal_distribution_comparison.ipynb` |

Some training notebooks contain alternative or historical cells. Follow the cell and path notes in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before using Run All. The numerical suffix of a checkpoint filename is an archive label; it should not automatically be interpreted as a standalone Hellinger score.