#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from complexity_core import (
    MISSING_VALUE_STRATEGIES,
    PYCOL_METRICS_NEED_UNNORM,
    PYCOL_METRICS_NO_DISTANCE,
    PYCOL_PRESETS_CLI_EPILOG,
    PycolMatrixMode,
    build_pycol_complexity,
    compute_pycol_metrics,
    get_all_pymfe_complexity_metrics,
    parse_pycol_metrics_selection,
    partition_pycol_metrics,
    resolve_pycol_matrix_mode,
    prepare_xy,
    subsample_xy_for_complexity,
)

CLI_VERSION = "1.7.0"
#
# Above this row count (after cleaning / subsampling), PyCol defaults to **sequential** metrics in one
# process via a single Complexity object — much lower peak RAM than a Process pool (each worker
# otherwise pickles X,y and builds its own full Complexity / distance structure).
PYCOL_SEQUENTIAL_ROW_THRESHOLD = 5_000


def extract_last_int(text: str) -> int | None:
    matches = re.findall(r"\d+", str(text))
    if not matches:
        return None
    return int(matches[-1])


def load_dataset(source: str, ref: str) -> tuple[pd.DataFrame, str]:
    ref = str(ref).strip()
    source_l = source.strip().lower()

    if source_l == "csv":
        p = Path(ref)
        if not p.exists():
            raise FileNotFoundError(f"CSV file not found: {p}")
        return pd.read_csv(p), p.stem

    if source_l == "uci":
        from ucimlrepo import fetch_ucirepo

        ds_id = extract_last_int(ref)
        if ds_id is None:
            raise ValueError(f"Could not parse UCI dataset id from: {ref}")
        ds = fetch_ucirepo(id=int(ds_id))
        x = ds.data.features
        y = ds.data.targets
        if y is None:
            raise ValueError("UCI dataset has no target column.")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        df = x.copy()
        df["target"] = y
        return df, f"uci_{ds_id}"

    if source_l == "openml":
        import openml

        ds_id = extract_last_int(ref)
        if ds_id is None:
            raise ValueError(f"Could not parse OpenML dataset id from: {ref}")
        ds = openml.datasets.get_dataset(int(ds_id))
        x, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        if y is None:
            raise ValueError("OpenML dataset has no default target attribute.")
        df = x.copy()
        df["target"] = y
        return df, f"openml_{ds_id}"

    raise ValueError("source must be one of: csv, uci, openml")


def _pycol_matrix_mode_for_metric(metric: str, run_mode: PycolMatrixMode) -> PycolMatrixMode:
    if metric in PYCOL_METRICS_NO_DISTANCE:
        return "skip"
    if metric in PYCOL_METRICS_NEED_UNNORM:
        return "both"
    return "both" if run_mode == "both" else "dist"


def pycol_metric_job(
    args: tuple[np.ndarray, np.ndarray, str, bool, int, PycolMatrixMode],
) -> tuple[str, Any]:
    x, y, metric, parallel_heom, heom_n_jobs, run_mode = args

    key = f"pycol_{metric}"
    mode = _pycol_matrix_mode_for_metric(metric, run_mode)
    try:
        comp = build_pycol_complexity(
            x,
            y,
            matrix_mode=mode,
            parallel_heom=parallel_heom and mode != "skip",
            heom_n_jobs=heom_n_jobs,
        )
        if not hasattr(comp, metric):
            return key, None
        val = getattr(comp, metric)()
        if isinstance(val, (float, int, np.floating, np.integer)):
            return key, float(val)
        arr = np.asarray(val, dtype=float)
        return key, (float(np.nanmean(arr)) if arr.size else None)
    except Exception:
        return key, None


def pymfe_metric_job(args: tuple[np.ndarray, np.ndarray, str]) -> tuple[str, Any]:
    x, y, metric = args
    from pymfe.mfe import MFE

    try:
        mfe = MFE(groups=["complexity"], features=[metric], summary=("mean",), random_state=0)
        mfe.fit(X=x, y=y)
        names, vals = mfe.extract()
        if not names:
            return f"pymfe_{metric}", None
        name = str(names[0])
        val = vals[0]
        key = f"pymfe_{name}"
        if isinstance(val, (float, int, np.floating, np.integer)):
            return key, float(val)
        arr = np.asarray(val, dtype=float)
        return key, (float(np.nanmean(arr)) if arr.size else None)
    except Exception:
        return f"pymfe_{metric}", None


def parse_metrics_arg(metrics_arg: str, library: str) -> list[str]:
    lib = library.lower()
    if lib == "pycol":
        names, _ = parse_pycol_metrics_selection(metrics_arg, custom_metrics=None)
        return names
    if metrics_arg.strip().lower() == "all":
        if lib == "pymfe":
            return list(get_all_pymfe_complexity_metrics())
        raise ValueError("For library=both, provide --pycol-metrics and --pymfe-metrics.")
    return [m.strip() for m in metrics_arg.split(",") if m.strip()]


def _progress_sink(enabled: bool, msg: str, *, end: str = "\n") -> None:
    if enabled:
        print(msg, end=end, file=sys.stderr, flush=True)


def run_parallel_jobs(
    library: str,
    x: np.ndarray,
    y: np.ndarray,
    metrics: list[str],
    n_jobs: int,
    *,
    show_progress: bool,
    desc: str,
    pycol_parallel_heom: bool = False,
    pycol_heom_n_jobs: int = 1,
    pycol_matrix_mode: PycolMatrixMode = "dist",
) -> dict[str, Any]:
    if not metrics:
        return {}
    if library == "pycol":
        worker: Callable[..., tuple[str, Any]] = pycol_metric_job
        run_mode: PycolMatrixMode = pycol_matrix_mode
        inputs = [
            (x, y, m, bool(pycol_parallel_heom), int(pycol_heom_n_jobs), run_mode)
            for m in metrics
        ]
    else:
        worker = pymfe_metric_job
        inputs = [(x, y, m) for m in metrics]
    total = len(inputs)
    # Never spawn more pool workers than tasks or CPUs — e.g. cheap=7 metrics with --n-jobs 70
    # would otherwise fork 70 processes per run and can exhaust resources across a batch.
    n_workers = max(1, min(int(n_jobs), total, max(1, mp.cpu_count() or 1)))
    if show_progress and n_workers < int(n_jobs):
        _progress_sink(
            True,
            f"      Capping pool to {n_workers} worker(s) (metrics={total}, "
            f"--n-jobs={int(n_jobs)}, cpus={mp.cpu_count()}).",
        )
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            use_tqdm = True
        except ImportError:
            use_tqdm = False

    with mp.Pool(processes=n_workers) as pool:
        if show_progress and use_tqdm:
            out = list(
                tqdm(
                    pool.imap_unordered(worker, inputs, chunksize=1),
                    total=total,
                    desc=desc,
                    unit="metric",
                    file=sys.stderr,
                    leave=True,
                )
            )
        elif show_progress:
            out = []
            for i, item in enumerate(pool.imap_unordered(worker, inputs, chunksize=1), start=1):
                out.append(item)
                pct = 100 * i // total if total else 100
                _progress_sink(
                    True,
                    f"\r{desc}: {i}/{total} metrics ({pct}%)",
                    end="" if i < total else "\n",
                )
        else:
            out = pool.map(worker, inputs)
    return {k: v for k, v in out}


def upsert_result_row(output_csv: Path, result: dict[str, Any], key_col: str = "dataset_name") -> pd.DataFrame:
    """
    Upsert one result row into output CSV.

    - If dataset exists (same key_col), update that row and add any new metric columns.
    - If dataset is new, append as a new row.
    """
    new_row_df = pd.DataFrame([result])

    if not output_csv.exists():
        return new_row_df

    existing = pd.read_csv(output_csv)
    if existing.empty:
        return new_row_df

    if key_col not in existing.columns:
        # Legacy file without key column: append preserving old rows.
        return pd.concat([existing, new_row_df], ignore_index=True, sort=False)

    key_val = result.get(key_col)
    mask = existing[key_col].astype(str) == str(key_val)
    if mask.any():
        idx = existing.index[mask][0]
        for col, val in result.items():
            if col not in existing.columns:
                existing[col] = np.nan
            existing.at[idx, col] = val
        return existing

    return pd.concat([existing, new_row_df], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel complexity runner for one dataset (CSV/UCI/OpenML).",
        epilog=PYCOL_PRESETS_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, choices=["csv", "uci", "openml"])
    parser.add_argument(
        "--ref",
        required=True,
        help="For csv: file path. For uci/openml: numeric dataset id (e.g. 53) or full URL; id is parsed from the string.",
    )
    parser.add_argument("--label-column", default="target", help="Label column for dataset.")
    parser.add_argument("--library", default="both", choices=["pycol", "pymfe", "both"])
    parser.add_argument(
        "--metrics",
        default="all",
        help=(
            "PyCol presets: cheap_minimal (no matrix, fast); cheap (all PyCol except T1/NSG/ICSV, one matrix max); "
            "expensive_core (T1, NSG, ICSV, two matrices); expensive; all; custom. "
            "Or comma-separated names (e.g. F1,N3,N4). See epilog below for why. pymfe: all | list."
        ),
    )
    parser.add_argument(
        "--pycol-metrics",
        default="all",
        help=(
            "When --library both: PyCol side — cheap_minimal | cheap | expensive_core | expensive | all | custom | or comma-separated names. "
            "If 'custom', also pass --pycol-custom-metrics."
        ),
    )
    parser.add_argument(
        "--pycol-custom-metrics",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated PyCol metric names when --metrics/--pycol-metrics is exactly 'custom' "
            "(example: F1,N1,N3,F1v)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {CLI_VERSION}",
    )
    parser.add_argument(
        "--pymfe-metrics",
        default="all",
        help="For library=both: pymfe metrics (all or comma-separated).",
    )
    parser.add_argument("--n-jobs", type=int, default=max(1, (mp.cpu_count() // 2)))
    parser.add_argument(
        "--missing-values",
        default="impute_median",
        choices=list(MISSING_VALUE_STRATEGIES),
        help=(
            "How to handle NaNs in feature columns after encoding: "
            "drop_rows removes any row with a remaining missing value; "
            "fill_zero / impute_median / impute_mean fill before modeling."
        ),
    )
    parser.add_argument("--output-csv", default="parallel_dataset_complexity.csv")
    parser.add_argument(
        "--complexity-max-rows",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If > 0 and n rows after encoding exceeds N, randomly subsample N rows (fixed seed) for "
            "PyCol/PyMFE only. Speeds large datasets; metrics are approximate on the subset. 0 = use all rows."
        ),
    )
    parser.add_argument(
        "--pycol-parallel-metrics",
        action="store_true",
        help=(
            "Use a process pool for PyCol (one worker per metric). Faster on small n, but on large n "
            "each worker duplicates the design matrix and can build its own distance structure — very high "
            f"peak RAM. Default: sequential PyCol in one process when n≥{PYCOL_SEQUENTIAL_ROW_THRESHOLD} "
            "(single Complexity instance, lower memory; still uses all rows unless you set "
            "--complexity-max-rows)."
        ),
    )
    parser.add_argument(
        "--pycol-distance-matrix",
        choices=("skip", "dist", "both", "auto", "build"),
        default="auto",
        help=(
            "HEOM storage (default: auto). auto: cheap_minimal→no matrix, cheap→one matrix, "
            "expensive_core→two matrices. dist/both/skip override. One matrix saves ~50%% RAM vs stock PyCol when N4,N2,… only."
        ),
    )
    parser.add_argument(
        "--pycol-parallel-heom",
        action="store_true",
        help=(
            "Parallel row workers when building the HEOM matrix (pycol_heom.py). "
            "Matrix build is always vectorized; this only adds multi-process rows. "
            "Use when HEOM tier is dist or both (not skip)."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar and step messages (for logs/CI).",
    )
    args = parser.parse_args()

    run_pycol = args.library in ("pycol", "both")
    if run_pycol:
        pycol_arg_chk = (
            args.metrics.strip().lower() if args.library == "pycol" else args.pycol_metrics.strip().lower()
        )
        if pycol_arg_chk == "custom":
            if not args.pycol_custom_metrics or not str(args.pycol_custom_metrics).strip():
                parser.error(
                    "When PyCol metrics are 'custom', pass --pycol-custom-metrics with a comma-separated list "
                    "(e.g. --pycol-custom-metrics N1,N3,F1,F1v)."
                )

    show_progress = not args.no_progress and sys.stderr.isatty()

    def step(msg: str) -> None:
        _progress_sink(show_progress, msg)

    run_pymfe = args.library in ("pymfe", "both")
    total_phases = 2 + int(run_pycol) + int(run_pymfe) + 1  # load, prepare, *metrics, save
    phase_i = 0

    def phase(msg: str) -> None:
        nonlocal phase_i
        phase_i += 1
        step(f"[{phase_i}/{total_phases}] {msg}")

    phase("Loading dataset …")
    df, dataset_name = load_dataset(args.source, args.ref)
    step(f"      Loaded `{dataset_name}`  shape={df.shape}")
    if args.label_column not in df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found. Available columns: {list(df.columns)}"
        )

    phase("Encoding features and labels …")
    x, y, _ = prepare_xy(df, label_col=args.label_column, missing_values=args.missing_values)
    step(
        f"      Ready: n_rows={x.shape[0]}  n_features={x.shape[1]}  n_classes={int(np.unique(y).size)}  "
        f"missing_values={args.missing_values!r}  n_jobs={max(1, args.n_jobs)}"
    )
    x_met, y_met, cmeta = subsample_xy_for_complexity(x, y, int(args.complexity_max_rows))
    if cmeta.get("complexity_subsampled"):
        step(
            f"      Complexity subsample: using {cmeta['n_rows_complexity_used']}/"
            f"{cmeta['n_rows_complexity_input']} rows (--complexity-max-rows)"
        )

    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "parallel_cli_version": CLI_VERSION,
        "source": args.source,
        "label_column": args.label_column,
        "missing_values": args.missing_values,
        "n_rows_original": int(df.shape[0]),
        "n_columns_original": int(df.shape[1]),
        "n_rows_used": int(x.shape[0]),
        "n_features_after_encoding": int(x.shape[1]),
        "n_classes": int(np.unique(y).size),
        "n_jobs": int(max(1, args.n_jobs)),
    }
    if int(args.complexity_max_rows) > 0:
        result["complexity_max_rows"] = int(args.complexity_max_rows)
    if cmeta.get("complexity_subsampled"):
        result["complexity_subsampled"] = True
        result["n_rows_complexity_input"] = int(cmeta["n_rows_complexity_input"])
        result["n_rows_complexity_used"] = int(cmeta["n_rows_complexity_used"])

    n_jobs = int(max(1, args.n_jobs))

    if run_pycol:
        pycol_arg = args.metrics if args.library == "pycol" else args.pycol_metrics
        pycol_metrics, pycol_preset = parse_pycol_metrics_selection(
            pycol_arg, custom_metrics=args.pycol_custom_metrics
        )
        result["pycol_metrics_preset"] = pycol_preset or "custom"
        pycol_matrix_mode = resolve_pycol_matrix_mode(
            pycol_metrics,
            preset=pycol_preset,
            override=args.pycol_distance_matrix,
        )
        result["pycol_matrix_mode"] = pycol_matrix_mode
        result["pycol_skip_distance_matrix"] = pycol_matrix_mode == "skip"
        no_dist_part, need_dist_omitted = partition_pycol_metrics(pycol_metrics)
        if pycol_matrix_mode == "skip":
            pycol_metrics_run = no_dist_part
            if need_dist_omitted and show_progress:
                step(
                    "      PyCol: Level A (skip) — omitting: "
                    + ", ".join(need_dist_omitted)
                )
        else:
            pycol_metrics_run = pycol_metrics
            if show_progress:
                step(f"      PyCol: HEOM tier `{pycol_matrix_mode}` for distance metrics.")
        n_pycol = int(x_met.shape[0])
        use_sequential_pycol = n_pycol >= PYCOL_SEQUENTIAL_ROW_THRESHOLD and not args.pycol_parallel_metrics
        if use_sequential_pycol:
            phase(
                f"PyCol: computing {len(pycol_metrics_run)} metric(s) sequentially (single Complexity, n={n_pycol}) …"
            )
            if show_progress:
                def _pycol_prog(m: str) -> None:
                    if m == "__init__":
                        _progress_sink(True, "      PyCol: initializing (no distance matrix) …")
                    elif m == "__init_dist__":
                        _progress_sink(True, "      PyCol: initializing (distance matrix) …")
                    else:
                        _progress_sink(True, f"      PyCol: `{m}` …")

                prog_cb = _pycol_prog
            else:
                prog_cb = None
            result.update(
                compute_pycol_metrics(
                    x_met,
                    y_met,
                    pycol_metrics_run,
                    matrix_mode=pycol_matrix_mode,
                    preset=pycol_preset,
                    parallel_heom=bool(args.pycol_parallel_heom)
                    and pycol_matrix_mode != "skip",
                    heom_n_jobs=n_jobs,
                    progress_callback=prog_cb,
                )
            )
            result["pycol_sequential_large_n"] = True
        else:
            phase(f"PyCol: computing {len(pycol_metrics_run)} metric(s) in parallel …")
            result.update(
                run_parallel_jobs(
                    "pycol",
                    x_met,
                    y_met,
                    pycol_metrics_run,
                    n_jobs,
                    show_progress=show_progress,
                    desc="pycol",
                    pycol_parallel_heom=bool(args.pycol_parallel_heom)
                    and pycol_matrix_mode != "skip",
                    pycol_heom_n_jobs=n_jobs,
                    pycol_matrix_mode=pycol_matrix_mode,
                )
            )
        step("      PyCol done.")

    if run_pymfe:
        pymfe_metrics = (
            parse_metrics_arg(args.metrics, "pymfe")
            if args.library == "pymfe"
            else parse_metrics_arg(args.pymfe_metrics, "pymfe")
        )
        phase(f"PyMFE: computing {len(pymfe_metrics)} metric(s) in parallel …")
        result.update(
            run_parallel_jobs(
                "pymfe",
                x_met,
                y_met,
                pymfe_metrics,
                n_jobs,
                show_progress=show_progress,
                desc="pymfe",
            )
        )
        step("      PyMFE done.")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase("Merging and writing CSV …")
    merged_df = upsert_result_row(out_path, result, key_col="dataset_name")
    merged_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}", file=sys.stderr if show_progress else sys.stdout, flush=True)
    print(f"Rows: {len(merged_df)}", file=sys.stderr if show_progress else sys.stdout, flush=True)
    print(f"Columns: {len(merged_df.columns)}", file=sys.stderr if show_progress else sys.stdout, flush=True)


if __name__ == "__main__":
    main()

