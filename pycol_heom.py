"""
Faster PyCol-compatible HEOM distance matrices.

Replaces the nested Python loops in ``pycol_complexity.Complexity.__distance_HEOM``
(see ``pycol-doc/complexity.py``) with vectorized NumPy row fills.

- Same formulas as PyCol (normalized + optional unnormalized n×n matrices).
- ``compute_unnorm=False`` skips the second matrix (~half RAM) when metrics only need ``dist_matrix``.
- Optional ``n_jobs > 1``: parallel row chunks (in-RAM matrices).
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
    *,
    compute_unnorm: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """HEOM distances from sample i to all rows (vectorized over features)."""
    n = int(x.shape[0])
    dist_sq = np.zeros(n, dtype=float)
    unnorm_sq = np.zeros(n, dtype=float) if compute_unnorm else None
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
            if compute_unnorm and unnorm_sq is not None:
                unnorm_sq += np.where(miss, 1.0, unnorm_contrib)
        else:
            mismatch = col != ai
            miss = np.isnan(col) | _feat_missing_scalar(ai)
            dist_sq += np.where(miss | mismatch, 1.0, 0.0)
            if compute_unnorm and unnorm_sq is not None:
                unnorm_sq += np.where(miss | mismatch, 1.0, 0.0)

    dist_sq[i] = 0.0
    d_row = np.sqrt(dist_sq)
    if not compute_unnorm:
        return d_row, None
    assert unnorm_sq is not None
    unnorm_sq[i] = 0.0
    return d_row, np.sqrt(unnorm_sq)


def _fill_matrix_rows(
    i_start: int,
    i_end: int,
    x: np.ndarray,
    meta: list[int],
    range_max: np.ndarray,
    range_min: np.ndarray,
    dist_matrix: np.ndarray,
    unnorm_dist_matrix: np.ndarray | None,
    *,
    compute_unnorm: bool,
    dtype: np.dtype,
) -> None:
    for i in range(i_start, i_end):
        d_row, ud_row = _heom_row_sq(
            i, x, meta, range_max, range_min, compute_unnorm=compute_unnorm
        )
        row = d_row.astype(dtype, copy=False)
        dist_matrix[i, :] = row
        dist_matrix[:, i] = row
        if compute_unnorm and unnorm_dist_matrix is not None and ud_row is not None:
            urow = ud_row.astype(dtype, copy=False)
            unnorm_dist_matrix[i, :] = urow
            unnorm_dist_matrix[:, i] = urow


def _heom_chunk_worker(
    args: tuple[int, int, np.ndarray, list[int], np.ndarray, np.ndarray, bool],
) -> tuple[int, int, np.ndarray, np.ndarray | None]:
    i_start, i_end, x, meta, range_max, range_min, compute_unnorm = args
    n = int(x.shape[0])
    dist_rows = np.zeros((i_end - i_start, n), dtype=float)
    unnorm_rows = (
        np.zeros((i_end - i_start, n), dtype=float) if compute_unnorm else None
    )
    for ii, i in enumerate(range(i_start, i_end)):
        d_row, ud_row = _heom_row_sq(
            i, x, meta, range_max, range_min, compute_unnorm=compute_unnorm
        )
        dist_rows[ii] = d_row
        if compute_unnorm and unnorm_rows is not None and ud_row is not None:
            unnorm_rows[ii] = ud_row
    return i_start, i_end, dist_rows, unnorm_rows


def build_heom_distance_matrices(
    x: np.ndarray,
    meta: list[int],
    *,
    n_jobs: int = 1,
    parallel_min_rows: int = _DEFAULT_PARALLEL_MIN_ROWS,
    compute_unnorm: bool = True,
    matrix_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``dist_matrix`` and optionally ``unnorm_dist_matrix`` (n×n), PyCol HEOM semantics."""
    dtype = np.dtype(matrix_dtype)
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        z = np.zeros((0, 0), dtype=dtype)
        return z, z.copy()

    meta_list = [int(m) for m in meta]
    range_max = np.max(x, axis=0)
    range_min = np.min(x, axis=0)

    dist_matrix = np.zeros((n, n), dtype=dtype)
    unnorm_dist_matrix: np.ndarray | None = (
        np.zeros((n, n), dtype=dtype) if compute_unnorm else None
    )

    use_parallel = int(n_jobs) > 1 and n >= int(parallel_min_rows)
    if not use_parallel:
        _fill_matrix_rows(
            0,
            n,
            x,
            meta_list,
            range_max,
            range_min,
            dist_matrix,
            unnorm_dist_matrix,
            compute_unnorm=compute_unnorm,
            dtype=dtype,
        )
    else:
        workers = max(1, min(int(n_jobs), n, os.cpu_count() or 1))
        chunk = max(1, (n + workers - 1) // workers)
        tasks: list[
            tuple[int, int, np.ndarray, list[int], np.ndarray, np.ndarray, bool]
        ] = []
        for w in range(workers):
            i_start = w * chunk
            i_end = min(n, i_start + chunk)
            if i_start >= i_end:
                continue
            tasks.append(
                (i_start, i_end, x, meta_list, range_max, range_min, compute_unnorm)
            )

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i_start, i_end, dist_rows, unnorm_rows in pool.map(_heom_chunk_worker, tasks):
                for ii, i in enumerate(range(i_start, i_end)):
                    row = dist_rows[ii].astype(dtype, copy=False)
                    dist_matrix[i, :] = row
                    dist_matrix[:, i] = row
                    if (
                        compute_unnorm
                        and unnorm_dist_matrix is not None
                        and unnorm_rows is not None
                    ):
                        urow = unnorm_rows[ii].astype(dtype, copy=False)
                        unnorm_dist_matrix[i, :] = urow
                        unnorm_dist_matrix[:, i] = urow

    if compute_unnorm and unnorm_dist_matrix is not None:
        return dist_matrix, unnorm_dist_matrix
    return dist_matrix, np.zeros((0, 0), dtype=dtype)
