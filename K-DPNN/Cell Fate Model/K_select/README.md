# R4：Model 1 的 K 值选择代码

本目录仅收录 R4 采用的独立 A/B/C 稳态窗口方案，包括代表参数选择、SDE 数据生成、GMM 容量评价、精度预算选型和对应绘图代码。输入数据、运行结果、历史实验和缓存均不随代码提供。

## 文件

| 文件 | 用途 |
|---|---|
| `prepare_representatives.py` | 从原训练参数中按标准化参数空间的 maximin 规则选择30组代表条件 |
| `select_k.py` | 生成独立 A/B/C 数据，拟合候选 GMM，评价 Hellinger 距离，汇总覆盖率与敏感性分析 |
| `density_utils.py` | 概率矩阵归一化、经验直方图、GMM 网格积分、Hellinger 距离及稳态初值处理 |
| `plot_k_selection.py` | 从逐条件评价记录重新计算并核对选型数值，绘制误差与覆盖率双面板图 |
| `requirements.txt` | 本代码使用的第三方依赖 |
| `.gitignore` | 排除数据、结果及缓存 |

## 实验设置

- 固定30组代表参数，候选 K 为3、4、5。
- 每个条件分别独立生成 A、B、C 三组数据；每组5000条轨迹，每条保留 `(80,100]` 内的200个时刻，共100万个二维状态点。
- A组全部样本用于完整协方差 GMM 拟合；B、C组分别构建独立参考概率矩阵。
- 条件级误差为相对B、C两份参考的 Hellinger 距离的算术平均。
- 选择误差不超过0.10的条件至少达到27/30时的最小候选 K；没有可行候选时报告无可行值。
- 模拟更新沿用原实验，包含漂移项、常规高斯随机增量、额外平方随机项和逐分量绝对值处理。采样时刻为80.1至100.0，不包含80.0。
- 同一轨迹内的时刻存在相关性；100万个状态点对应5000条独立轨迹。

## 输入数据

运行时需要另外提供原实验的两个输入文件：

1. `ps_train.csv`：含表头的四列训练参数，依次为自激活强度、相互抑制强度、降解率、噪声强度。
2. `train_all_xstable_and_w.mat`：包含与参数表逐行对应的 `all_xstable` 和 `all_w`，用于稳态初值采样。

本代码读取预先计算的稳定点与权重；稳定点求解程序不在这份代码中。所有输入路径均通过命令行提供，不依赖原项目目录。

## 运行

使用 Python 3.10 或更高版本。安装依赖后，在本目录执行以下命令；将示例中的 `path/to` 替换为实际输入文件位置。

```text
python -m pip install -r requirements.txt

python prepare_representatives.py --parameters "path/to/ps_train.csv" --stable-states "path/to/train_all_xstable_and_w.mat" --output "data/representative_parameters.csv"

python select_k.py generate --representatives "data/representative_parameters.csv" --stable-states "path/to/train_all_xstable_and_w.mat"

python select_k.py evaluate

python plot_k_selection.py
```

`generate` 和 `evaluate` 默认输出到本目录的 `results/`。评价阶段默认读取该目录中由生成阶段保存的代表参数副本。绘图阶段默认读取 `results/`，将图片输出到 `results/figures/`。

如使用其他结果位置，分别给生成和评价命令传入相同的 `--output-dir`，给绘图命令传入对应的 `--results-dir`。计算资源可通过 `--threads`、`--chunk-trajectories`、`--workers` 调整；用 `--help` 查看各入口参数。

生成与拟合保留原有断点续跑逻辑。更换输入、拟合设置或代码后应使用新的输出目录。

## 代码来源与整理范围

- `select_k.py` 来自 `12_window_abc_million_coverage.py`；整理了输入输出路径和辅助函数导入，保留原数值计算与选型规则。
- maximin 选点函数来自 `06_generate_maximin_million_endpoints.py`；不包含其旧终点采样实验。
- 四个概率计算函数来自 `09_abc_soft_threshold.py`，仅提取当前主流程实际调用的函数；不包含软阈值实验入口。
- 绘图代码按原独立 A/B/C 选型图整理，所选 K 从输入结果计算，不写死为某个值。
- 不包含 BIC、样本量探索、旧样本池重划分、千万点试运行、网络训练、时间对比或它们的运行结果。
