#!/usr/bin/env python3
"""
Compare PyCol native HEOM distance matrices vs project build (pycol_heom.py).

Usage:
  python validate_pycol_heom.py --synthetic --n-rows 200
  python validate_pycol_heom.py --uci-id 186 --n-rows 500
  python validate_pycol_heom.py --source uci --ref 17 --label-column target

Exit 0 if both dist_matrix and unnorm_dist_matrix match within --atol; else exit 1.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from complexity_core import (
    _evaluate_pycol_metric,
    _init_pycol_complexity_shell,
    prepare_xy,
    subsample_xy_for_complexity,
)
from parallel_complexity_cli import load_dataset
from pycol_heom import build_heom_distance_matrices


def _native_heom_matrices(x: np.ndarray, meta: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """PyCol installed package: Complexity.__distance_HEOM (nested loops)."""
    from pycol_complexity import complexity as pycol_complexity

    comp = pycol_complexity.Complexity.__new__(pycol_complexity.Complexity)
    comp.meta = [int(m) for m in meta]
    x_f = np.asarray(x, dtype=float)
    return comp._Complexity__distance_HEOM(x_f)


def _matrix_diff_report(
    name: str,
    native: np.ndarray,
    fast: np.ndarray,
    *,
    atol: float,
) -> tuple[bool, dict[str, float]]:
    diff = np.abs(native - fast)
    ok = bool(np.allclose(native, fast, rtol=0.0, atol=atol))
    report = {
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.nanmean(diff)) if diff.size else 0.0,
    }
    status = "PASS" if ok else "FAIL"
    print(
        f"  {name}: {status}  max_abs_diff={report['max_abs_diff']:.6e}  "
        f"mean_abs_diff={report['mean_abs_diff']:.6e}  (atol={atol})"
    )
    if not ok:
        i, j = np.unravel_index(int(np.nanargmax(diff)), diff.shape)
        print(f"    worst pair (i,j)=({i},{j})  native={native[i,j]:.12g}  build={fast[i,j]:.12g}")
    return ok, report


def _optional_n3_check(
    x: np.ndarray,
    y: np.ndarray,
    dist_native: np.ndarray,
    dist_fast: np.ndarray,
    unnorm_native: np.ndarray,
    unnorm_fast: np.ndarray,
) -> None:
    from pycol_complexity import complexity as pycol_complexity

    y_a = np.asarray(y)
    comp_nat = _init_pycol_complexity_shell(
        pycol_complexity, x, y_a, dist_matrix=dist_native, unnorm_dist_matrix=unnorm_native
    )
    comp_fast = _init_pycol_complexity_shell(
        pycol_complexity, x, y_a, dist_matrix=dist_fast, unnorm_dist_matrix=unnorm_fast
    )
    n3_nat = _evaluate_pycol_metric(comp_nat, "N3")
    n3_fast = _evaluate_pycol_metric(comp_fast, "N3")
    if n3_nat is None or n3_fast is None:
        print("  N3: skipped (metric unavailable)")
        return
    match = n3_nat == n3_fast or (
        np.isfinite(n3_nat)
        and np.isfinite(n3_fast)
        and abs(n3_nat - n3_fast) < 1e-12
    )
    tag = "PASS" if match else "FAIL"
    print(f"  N3 (sanity): {tag}  native={n3_nat}  build={n3_fast}")


def _synthetic_xy(n_rows: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, n_features)).astype(np.float64)
    y = rng.integers(0, 3, size=n_rows)
    return x, y


def run_validation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    atol: float,
    check_parallel: bool,
    parallel_jobs: int,
) -> bool:
    from pycol_complexity import complexity as pycol_complexity

    x_f = np.asarray(x, dtype=float)
    n, p = x_f.shape
    comp = pycol_complexity.Complexity.__new__(pycol_complexity.Complexity)
    meta = comp.is_categorical(np.array(x_f))

    print(f"HEOM matrix validation  n={n:,}  features={p}  meta={meta}")

    dist_native, unnorm_native = _native_heom_matrices(x_f, meta)
    dist_fast, unnorm_fast = build_heom_distance_matrices(
        x_f, meta, n_jobs=1, compute_unnorm=True
    )
    dist_only, unnorm_empty = build_heom_distance_matrices(
        x_f, meta, n_jobs=1, compute_unnorm=False
    )
    ok_dist_only, _ = _matrix_diff_report(
        "dist_matrix (dist-only build vs full)", dist_native, dist_only, atol=atol
    )
    if unnorm_empty.size != 0:
        print("  unnorm skipped: FAIL  expected empty (0,0) matrix")
        ok_unnorm_skip = False
    else:
        print("  unnorm skipped: PASS  empty (0,0) stored")
        ok_unnorm_skip = True

    ok_dist, _ = _matrix_diff_report("dist_matrix", dist_native, dist_fast, atol=atol)
    ok_unnorm, _ = _matrix_diff_report(
        "unnorm_dist_matrix", unnorm_native, unnorm_fast, atol=atol
    )
    _optional_n3_check(x_f, y, dist_native, dist_fast, unnorm_native, unnorm_fast)

    ok_parallel = True
    if check_parallel and n >= 2:
        dist_par, unnorm_par = build_heom_distance_matrices(
            x_f, meta, n_jobs=max(2, parallel_jobs)
        )
        ok_p1, _ = _matrix_diff_report(
            "dist_matrix (parallel vs serial build)", dist_fast, dist_par, atol=atol
        )
        ok_p2, _ = _matrix_diff_report(
            "unnorm_dist_matrix (parallel vs serial build)", unnorm_fast, unnorm_par, atol=atol
        )
        ok_parallel = ok_p1 and ok_p2

    return ok_dist and ok_unnorm and ok_dist_only and ok_unnorm_skip and ok_parallel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify pycol_heom build matrices match PyCol native __distance_HEOM."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random numeric data (no download).",
    )
    parser.add_argument("--synthetic-rows", type=int, default=120, help="Rows for --synthetic.")
    parser.add_argument("--synthetic-cols", type=int, default=8, help="Features for --synthetic.")
    parser.add_argument(
        "--uci-id",
        type=int,
        default=None,
        metavar="ID",
        help="Shorthand: --source uci --ref ID --label-column target",
    )
    parser.add_argument("--source", choices=["csv", "uci", "openml"], default="uci")
    parser.add_argument("--ref", default="", help="Dataset ref (id or URL/path).")
    parser.add_argument("--label-column", default="target")
    parser.add_argument(
        "--missing-values",
        default="impute_median",
        choices=["drop_rows", "fill_zero", "impute_median", "impute_mean"],
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=0,
        help="Subsample to at most this many rows (0 = all rows).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Max absolute element-wise difference allowed.",
    )
    parser.add_argument(
        "--check-parallel",
        action="store_true",
        help="Also compare serial vs parallel pycol_heom build.",
    )
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0, help="Subsample RNG seed.")
    args = parser.parse_args(argv)

    if args.synthetic:
        x, y = _synthetic_xy(args.synthetic_rows, args.synthetic_cols, args.seed)
        label = "synthetic"
    else:
        ref = str(args.ref).strip()
        if args.uci_id is not None:
            ref = str(args.uci_id)
            args.source = "uci"
        if not ref:
            parser.error("Provide --synthetic, --uci-id ID, or --ref with --source.")
        df, label = load_dataset(args.source, ref)
        x, y, _ = prepare_xy(df, args.label_column, missing_values=args.missing_values)
        if args.n_rows > 0:
            x, y, meta = subsample_xy_for_complexity(x, y, args.n_rows, random_state=args.seed)
            if meta.get("complexity_subsampled"):
                print(f"Subsampled to n={meta['n_rows_complexity_used']:,} (seed={args.seed})")

    ok = run_validation(
        x,
        y,
        atol=float(args.atol),
        check_parallel=bool(args.check_parallel),
        parallel_jobs=int(args.parallel_jobs),
    )
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}  dataset={label}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
