# Reproducibility notes

These notes describe the actual source files in this repository snapshot. Adding this documentation did not retrain models, regenerate complete datasets, or change the existing numerical algorithms.

## Simulation protocols

Both Cell Fate workflows retain 200 sample times in `(80,100]`, but they currently use different random updates:

| Component | Additive random update per coordinate, before nonnegative reflection |
|---|---|
| Revised `data_generate/Simulation_density_*.ipynb` | `sqrt(2 * D * delta) * xi` |
| Archived `K_select/select_k.py` | `D * sqrt(delta) * r0 + 0.5 * delta * D**2 * (r1**2 - 1)` |

Here `xi`, `r0`, and `r1` are standard normal variables, with the archived script drawing independent variables for the two terms. Consequently, the same numerical value of `D` has a different role across the two components. The K-selection code preserves the archived independent A/B/C experiment; documentation of the revised Euler–Maruyama generator should not be interpreted as validation of that archived experiment under the new update.

An aligned end-to-end experiment requires one stated simulation convention, newly generated inputs where necessary, and re-evaluation of K selection and model performance. The included checkpoint files remain archived models.

## Notebook execution and historical cells

### Cell Fate

- In `Train_load_model.ipynb`, the first cell starting with `#完整训练模块，运行直接训练` explicitly creates a K=3 network. A later module cell starting with `#训练模块代码，不带主程序` creates K=4, followed by a separate `#训练主程序` cell. For the intended K=4 training path, use the later module and its matching main cell; do not run both alternative training paths indiscriminately.
- The final plotting cell in that training notebook loads `best_model_256-512-256.pth`, a training-output name that differs from the supplied archived checkpoint name. Set the path to the intended checkpoint before plotting.
- The first cell of `Plot_compare_save.ipynb` contains an uncommented Chinese description and color-map notes. Its first line is not valid Python. Treat that cell as documentation, convert it to Markdown, or comment the prose before using Run All.
- `data_generate/Simulation_density_train.ipynb` is currently configured with `n_cells = 10`; the validation and test generators use 50,000. The retained state count is `n_cells * 200` per parameter condition, not the number of independent trajectories.

### Coupled Toggle-Switch

- `Train_plot.ipynb` includes the training implementation, a main configuration cell, and a plotting cell. Its intended configuration is input dimension 5, K=4, and hidden layers 512–1024–512.
- Provide all three density datasets before training. Parameter tables and the main checkpoint are present, but the density matrices are absent.

### MESC

- The main configuration in `retrain_save.ipynb` sets input dimension 4 and K=3. The model class also has historical defaults for a different configuration; use the explicit MESC configuration.
- Its training main cell refers to missing `*_filtered` parameter tables and density directories. The dataset's filtering rule must be recovered before using that configuration.
- Its plotting cells refer to `saved_models_10runs_DRL/...run11...pth`, while this snapshot supplies a checkpoint directly under `train_model/`. Use the supplied checkpoint path only after confirming it is the intended model.
- The separate landscape, marginal, and statistics notebooks point to the main checkpoint filename in `train_model`, but still require the missing MESC parameter tables and density matrices.

## Scope of the supplied archive

This snapshot does not include complete ten-run training archives, per-condition density datasets, all paper timing/benchmark programs, or all source data for R4 figures and tables. The current files therefore support the workflows whose inputs are supplied separately; they are not a complete standalone reproduction bundle for every R4 result.

Most notebooks retain historical outputs. Those outputs are records of earlier executions and do not prove that the present code and current local inputs have been run together. Use a fresh kernel and a documented configuration for new calculations.

Before a final research release, record the exact package versions, model/dataset identifiers, input hashes, split definitions, random seeds, projection conventions, and the distinction between checkpoint-selection loss and reported Hellinger distance. Add the final citation, data location, and author-selected license when available.
