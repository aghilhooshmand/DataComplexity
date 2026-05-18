"""
HEOM pairwise distance matrices (PyCol-compatible).

Logic matches ``pycol_complexity.complexity.Complexity.__distance_HEOM`` (see
``pycol-doc/complexity.py`` and the installed package). Use instead of PyCol's
sequential ``__init__`` matrix build via :func:`build_heom_distance_matrices`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

# Rows below this: always sequential (process spawn overhead dominates).
_DEFAULT_PARALLEL_MIN_ROWS = 256


def _feat_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def heom_pair_distances(
    row_i: np.ndarray,
    row_j: np.ndarray,
    meta: list[int],
    range_max: np.ndarray,
    range_min: np.ndarray,
) -> tuple[float, float]:
    """Squared HEOM terms for one pair; returns (sqrt(norm), sqrt(unnorm))."""
    dist = 0.0
    unnorm_dist = 0.0
    for k, mk in enumerate(meta):
        ai, aj = row_i[k], row_j[k]
        if _feat_missing(ai) or _feat_missing(aj):
            dist += 1.0
            unnorm_dist += 1.0
            continue
        if mk == 0:
            diff = abs(float(ai) - float(aj))
            if range_max[k] == range_min[k]:
                dist += diff**2
                unnorm_dist += diff**2
            else:
                span = float(range_max[k] - range_min[k])
                dist += (diff / span) ** 2
                unnorm_dist += diff**2
        elif mk == 1:
            if ai != aj:
                dist += 1.0
                unnorm_dist += 1.0
    return float(np.sqrt(dist)), float(np.sqrt(unnorm_dist))


def _heom_row_pairs(
    i: int,
    x: np.ndarray,
    meta: list[int],
    range_max: np.ndarray,
    range_min: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    n = int(x.shape[0])
    out: list[tuple[int, int, float, float]] = []
    row_i = x[i]
    for j in range(i + 1, n):
        d, ud = heom_pair_distances(row_i, x[j], meta, range_max, range_min)
        out.append((i, j, d, ud))
    return out


def _heom_chunk_worker(
    args: tuple[int, int, np.ndarray, list[int], np.ndarray, np.ndarray],
) -> list[tuple[int, int, float, float]]:
    i_start, i_end, x, meta, range_max, range_min = args
    pairs: list[tuple[int, int, float, float]] = []
    for i in range(i_start, i_end):
        pairs.extend(_heom_row_pairs(i, x, meta, range_max, range_min))
    return pairs


def build_heom_distance_matrices(
    x: np.ndarray,
    meta: list[int],
    *,
    n_jobs: int = 1,
    parallel_min_rows: int = _DEFAULT_PARALLEL_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ``dist_matrix`` and ``unnorm_dist_matrix`` (n×n), PyCol HEOM semantics.

    Parameters
    ----------
    n_jobs:
        If > 1 and ``n >= parallel_min_rows``, fill the upper triangle in parallel
        (row chunks). Otherwise sequential.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        z = np.zeros((0, 0), dtype=float)
        return z, z.copy()

    meta_list = [int(m) for m in meta]
    range_max = np.max(x, axis=0)
    range_min = np.min(x, axis=0)

    dist_matrix = np.zeros((n, n), dtype=float)
    unnorm_dist_matrix = np.zeros((n, n), dtype=float)

    use_parallel = int(n_jobs) > 1 and n >= int(parallel_min_rows)
    if not use_parallel:
        for i in range(n):
            for tup in _heom_row_pairs(i, x, meta_list, range_max, range_min):
                i, j, d, ud = tup
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                unnorm_dist_matrix[i, j] = ud
                unnorm_dist_matrix[j, i] = ud
        return dist_matrix, unnorm_dist_matrix

    workers = max(1, min(int(n_jobs), n, os.cpu_count() or 1))
    chunk = max(1, (n + workers - 1) // workers)
    tasks: list[tuple[int, int, np.ndarray, list[int], np.ndarray, np.ndarray]] = []
    for w in range(workers):
        i_start = w * chunk
        i_end = min(n, i_start + chunk)
        if i_start >= i_end:
            continue
        tasks.append((i_start, i_end, x, meta_list, range_max, range_min))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk_pairs in pool.map(_heom_chunk_worker, tasks):
            for i, j, d, ud in chunk_pairs:
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                unnorm_dist_matrix[i, j] = ud
                unnorm_dist_matrix[j, i] = ud

    return dist_matrix, unnorm_dist_matrix
