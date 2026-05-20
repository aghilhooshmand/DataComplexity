"""
Faster PyCol-compatible HEOM distance matrices.

Replaces the nested Python loops in ``pycol_complexity.Complexity.__distance_HEOM``
(see ``pycol-doc/complexity.py``) with vectorized NumPy row fills.

- Same formulas as PyCol (normalized + unnormalized n×n matrices).
- Still stores both matrices in RAM (PyCol N2, N3, C1, … need ``comp.dist_matrix``).
- Optional ``n_jobs > 1``: parallel row chunks via :func:`build_heom_distance_matrices`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

_DEFAULT_PARALLEL_MIN_ROWS = 256


def _feat_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _heom_row_sq(
    i: int,
    x: np.ndarray,
    meta: list[int],
    range_max: np.ndarray,
    range_min: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """HEOM distances from sample i to all rows (vectorized over features)."""
    n = int(x.shape[0])
    dist_sq = np.zeros(n, dtype=float)
    unnorm_sq = np.zeros(n, dtype=float)
    row_i = x[i]

    for k, mk in enumerate(meta):
        col = x[:, k]
        ai = row_i[k]
        if mk == 0:
            diff = np.abs(col - ai)
            miss = np.isnan(col) | _feat_missing_scalar(ai)
            span = float(range_max[k] - range_min[k])
            if span == 0.0:
                contrib = diff**2
                unnorm_contrib = diff**2
            else:
                contrib = (diff / span) ** 2
                unnorm_contrib = diff**2
            dist_sq += np.where(miss, 1.0, contrib)
            unnorm_sq += np.where(miss, 1.0, unnorm_contrib)
        else:
            mismatch = col != ai
            miss = np.isnan(col) | _feat_missing_scalar(ai)
            dist_sq += np.where(miss | mismatch, 1.0, 0.0)
            unnorm_sq += np.where(miss | mismatch, 1.0, 0.0)

    dist_sq[i] = 0.0
    unnorm_sq[i] = 0.0
    return np.sqrt(dist_sq), np.sqrt(unnorm_sq)


def _fill_matrix_rows(
    i_start: int,
    i_end: int,
    x: np.ndarray,
    meta: list[int],
    range_max: np.ndarray,
    range_min: np.ndarray,
    dist_matrix: np.ndarray,
    unnorm_dist_matrix: np.ndarray,
) -> None:
    for i in range(i_start, i_end):
        d_row, ud_row = _heom_row_sq(i, x, meta, range_max, range_min)
        dist_matrix[i, :] = d_row
        dist_matrix[:, i] = d_row
        unnorm_dist_matrix[i, :] = ud_row
        unnorm_dist_matrix[:, i] = ud_row


def _heom_chunk_worker(
    args: tuple[int, int, np.ndarray, list[int], np.ndarray, np.ndarray],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    i_start, i_end, x, meta, range_max, range_min = args
    n = int(x.shape[0])
    dist_rows = np.zeros((i_end - i_start, n), dtype=float)
    unnorm_rows = np.zeros((i_end - i_start, n), dtype=float)
    for ii, i in enumerate(range(i_start, i_end)):
        d_row, ud_row = _heom_row_sq(i, x, meta, range_max, range_min)
        dist_rows[ii] = d_row
        unnorm_rows[ii] = ud_row
    return i_start, i_end, dist_rows, unnorm_rows


def build_heom_distance_matrices(
    x: np.ndarray,
    meta: list[int],
    *,
    n_jobs: int = 1,
    parallel_min_rows: int = _DEFAULT_PARALLEL_MIN_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ``dist_matrix`` and ``unnorm_dist_matrix`` (n×n), PyCol HEOM semantics.
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
        _fill_matrix_rows(0, n, x, meta_list, range_max, range_min, dist_matrix, unnorm_dist_matrix)
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
        for i_start, i_end, dist_rows, unnorm_rows in pool.map(_heom_chunk_worker, tasks):
            for ii, i in enumerate(range(i_start, i_end)):
                dist_matrix[i, :] = dist_rows[ii]
                dist_matrix[:, i] = dist_rows[ii]
                unnorm_dist_matrix[i, :] = unnorm_rows[ii]
                unnorm_dist_matrix[:, i] = unnorm_rows[ii]

    return dist_matrix, unnorm_dist_matrix
