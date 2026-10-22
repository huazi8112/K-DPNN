"""R4独立A/B/C方案使用的概率网格、Hellinger距离与稳态初值辅助函数。

保留原始数值实现；不包含历史软阈值实验入口。
"""
from __future__ import annotations

import numpy as np


def normalized_density(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("概率矩阵不能为空")
    density = matrix / total
    density = density + epsilon
    return density / density.sum()


def hellinger(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.sqrt(0.5 * np.square(np.sqrt(first) - np.sqrt(second)).sum())
    )


def empirical_density(
    points: np.ndarray,
    edges: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    count, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(edges, edges),
    )
    coverage = float(count.sum() / len(points))
    return normalized_density(count, epsilon), coverage


def gmm_grid_mass(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, float]:
    """通过各高斯分量在网格角点的二维CDF计算单元概率质量。"""
    from scipy.stats import multivariate_normal

    grid_x, grid_y = np.meshgrid(edges, edges, indexing="ij")
    corners = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    corner_shape = grid_x.shape
    mass = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    for weight, mean, covariance in zip(weights, means, covariances):
        cdf = multivariate_normal.cdf(
            corners,
            mean=mean,
            cov=covariance,
            maxpts=1_000_000,
            abseps=1e-8,
            releps=1e-8,
        ).reshape(corner_shape)
        component = (
            cdf[1:, 1:]
            - cdf[:-1, 1:]
            - cdf[1:, :-1]
            + cdf[:-1, :-1]
        )
        # CDF数值积分可能产生约机器精度量级的负数。
        mass += float(weight) * np.clip(component, 0.0, None)
    coverage = float(mass.sum())
    return mass, coverage


def stable_distribution(states, weights) -> tuple[np.ndarray, np.ndarray]:
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    if states.shape[1] != 2 and states.shape[0] == 2:
        states = states.T
    weights = np.asarray(weights, dtype=np.float64).ravel()
    valid = np.isfinite(states).all(axis=1) & np.isfinite(weights) & (weights > 0)
    states, weights = states[valid], weights[valid]
    if not len(states) or len(states) != len(weights):
        raise ValueError("稳态初值及权重无效")
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    return np.ascontiguousarray(states), np.ascontiguousarray(cdf)
