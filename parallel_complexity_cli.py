#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import traceback
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
        return pd.read_csv(p), p.name

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
    args: tuple[np.ndarray, np.ndarray, str, bool, int, PycolMatrixMode, np.dtype],
) -> tuple[str, Any]:
    x, y, metric, parallel_heom, heom_n_jobs, run_mode, matrix_dtype = args

    key = f"pycol_{metric}"
    mode = _pycol_matrix_mode_for_metric(metric, run_mode)
    try:
        comp = build_pycol_complexity(
            x,
            y,
            matrix_mode=mode,
            parallel_heom=parallel_heom and mode != "skip",
            heom_n_jobs=heom_n_jobs,
            matrix_dtype=matrix_dtype,
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


def make_pycol_progress_callback(
    enabled: bool,
    *,
    initial_completed: int = 0,
) -> Callable[[str, int, int], None] | None:
    """Per-metric CLI progress: start/done lines plus an optional tqdm bar."""
    if not enabled:
        return None

    pbar_holder: dict[str, Any] = {"bar": None}

    def _log(msg: str) -> None:
        bar = pbar_holder.get("bar")
        if bar is not None:
            try:
                from tqdm import tqdm  # type: ignore[import-untyped]

                tqdm.write(msg, file=sys.stderr)
            except ImportError:
                _progress_sink(True, msg)
        else:
            _progress_sink(True, msg)

    def _ensure_bar(total: int) -> None:
        if pbar_holder["bar"] is not None or total <= 0:
            return
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            pbar_holder["bar"] = tqdm(
                total=total,
                initial=initial_completed,
                desc="PyCol metrics",
                file=sys.stderr,
                unit="metric",
                leave=True,
            )
        except ImportError:
            pass

    def cb(event: str, index: int, total: int) -> None:
        if event == "__init__":
            _log("      PyCol: fast metrics (no distance matrix) …")
            return
        if event == "__init_dist__":
            bar = pbar_holder.get("bar")
            if bar is not None:
                bar.close()
                pbar_holder["bar"] = None
            _log(
                "      PyCol: building HEOM distance matrix (can take hours on large n) …",
            )
            return
        if event.startswith("done:"):
            metric = event[5:]
            remaining = max(0, total - index)
            _ensure_bar(total)
            bar = pbar_holder.get("bar")
            if bar is not None:
                bar.update(1)
            _log(
                f"      PyCol [{index}/{total}] `{metric}` done — {remaining} remaining",
            )
            return

        remaining = max(0, total - index + 1)
        _ensure_bar(total)
        _log(f"      PyCol [{index}/{total}] `{event}` … ({remaining} to go)")

    return cb


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
    pycol_matrix_dtype: np.dtype | type = np.float64,
    on_metric_complete: Callable[[str, Any], None] | None = None,
) -> dict[str, Any]:
    if not metrics:
        return {}
    if library == "pycol":
        worker: Callable[..., tuple[str, Any]] = pycol_metric_job
        run_mode: PycolMatrixMode = pycol_matrix_mode
        dtype = np.dtype(pycol_matrix_dtype)
        inputs = [
            (x, y, m, bool(pycol_parallel_heom), int(pycol_heom_n_jobs), run_mode, dtype)
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
            out = []
            for item in tqdm(
                pool.imap_unordered(worker, inputs, chunksize=1),
                total=total,
                desc=desc,
                unit="metric",
                file=sys.stderr,
                leave=True,
            ):
                out.append(item)
                key, val = item
                if on_metric_complete is not None and library == "pycol" and key.startswith("pycol_"):
                    on_metric_complete(key[6:], val)
        elif show_progress:
            out = []
            for i, item in enumerate(pool.imap_unordered(worker, inputs, chunksize=1), start=1):
                out.append(item)
                key, val = item
                if on_metric_complete is not None and library == "pycol" and key.startswith("pycol_"):
                    on_metric_complete(key[6:], val)
                pct = 100 * i // total if total else 100
                _progress_sink(
                    True,
                    f"\r{desc}: {i}/{total} metrics ({pct}%)",
                    end="" if i < total else "\n",
                )
        else:
            out = []
            for item in pool.map(worker, inputs):
                out.append(item)
                key, val = item
                if on_metric_complete is not None and library == "pycol" and key.startswith("pycol_"):
                    on_metric_complete(key[6:], val)
    return {k: v for k, v in out}


def _is_boolish(val: Any) -> bool:
    return isinstance(val, (bool, np.bool_))


def _coerce_value_for_column(val: Any, series: pd.Series) -> Any:
    """Make ``val`` storable in ``series`` without pandas dtype errors."""
    if pd.api.types.is_bool_dtype(series.dtype):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return val
        return bool(val)
    if _is_boolish(val):
        if pd.api.types.is_numeric_dtype(series.dtype):
            return float(int(val))
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _ensure_column_for_upsert(df: pd.DataFrame, col: str, sample_val: Any) -> None:
    """Add a column if missing, compatible with legacy float64 summary CSVs."""
    if col in df.columns:
        return
    if _is_boolish(sample_val):
        df[col] = np.nan
    elif isinstance(sample_val, (int, float, np.integer, np.floating)) and not _is_boolish(
        sample_val
    ):
        df[col] = np.nan
    else:
        df[col] = None


def _prepare_result_row_for_upsert(result: dict[str, Any], existing: pd.DataFrame) -> dict[str, Any]:
    """Coerce values so upsert works against legacy float64 columns."""
    prepared: dict[str, Any] = {}
    for col, val in result.items():
        if col in existing.columns:
            prepared[col] = _coerce_value_for_column(val, existing[col])
        elif _is_boolish(val):
            prepared[col] = float(int(val))
        else:
            prepared[col] = val
    return prepared


def _stored_value_is_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


def pycol_metric_is_done(existing_row: dict[str, Any], metric: str) -> bool:
    return _stored_value_is_present(existing_row.get(f"pycol_{metric}"))


def load_dataset_result_row(output_csv: Path, key_col: str, key_val: str) -> dict[str, Any]:
    """Return the saved CSV row for one dataset, or {} if not found."""
    if not output_csv.exists():
        return {}
    existing = pd.read_csv(output_csv)
    if existing.empty or key_col not in existing.columns:
        return {}
    mask = existing[key_col].astype(str) == str(key_val)
    if not mask.any():
        return {}
    row = existing.loc[mask].iloc[0]
    return {str(k): (None if pd.isna(v) else v) for k, v in row.items()}


def save_result_checkpoint(
    output_csv: Path,
    result: dict[str, Any],
    *,
    key_col: str,
) -> None:
    """Upsert one result row and write CSV immediately (per-metric checkpoint)."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged = upsert_result_row(output_csv, result, key_col=key_col)
    merged.to_csv(output_csv, index=False)


def make_pycol_checkpoint_saver(
    output_csv: Path,
    result: dict[str, Any],
    *,
    key_col: str,
) -> Callable[[str, Any], None]:
    """Save each completed PyCol metric to the output CSV as soon as it finishes."""

    def _save(metric: str, value: Any) -> None:
        result[f"pycol_{metric}"] = value
        result.pop("error", None)
        result.pop("error_traceback", None)
        save_result_checkpoint(output_csv, result, key_col=key_col)

    return _save


def append_failure_log(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message)
        if not message.endswith("\n"):
            fh.write("\n")


def build_error_result(
    args: argparse.Namespace,
    exc: BaseException,
    *,
    dataset_name: str | None = None,
    dataset_file: str | None = None,
) -> dict[str, Any]:
    ds_name = dataset_name or (Path(args.ref).stem if args.source == "csv" else str(args.ref))
    ds_file = dataset_file
    if ds_file is None:
        if args.source == "csv":
            ds_file = Path(args.ref).name
        else:
            ds_file = ds_name if str(ds_name).endswith(".csv") else f"{ds_name}.csv"
    tb = traceback.format_exc()
    return {
        "dataset_name": ds_name,
        "dataset_file": ds_file,
        "parallel_cli_version": CLI_VERSION,
        "source": args.source,
        "label_column": args.label_column,
        "missing_values": args.missing_values,
        "n_jobs": int(max(1, args.n_jobs)),
        "error": f"{type(exc).__name__}: {exc}",
        "error_traceback": tb,
    }


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
    row = _prepare_result_row_for_upsert(result, existing)
    if mask.any():
        idx = existing.index[mask][0]
        for col, val in row.items():
            _ensure_column_for_upsert(existing, col, val)
            if col in existing.columns:
                val = _coerce_value_for_column(val, existing[col])
            existing.at[idx, col] = val
        return existing

    new_row_df = pd.DataFrame([row])
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
        "--upsert-key",
        default="dataset_name",
        choices=("dataset_name", "dataset_file"),
        help=(
            "Column used to merge rows in --output-csv. "
            "Use dataset_file when appending to pmlb_DS/datasets_complexity_summary.csv (values like iris.csv)."
        ),
    )
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
        "--pycol-matrix-dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Storage dtype for n×n HEOM matrices (float64 matches stock PyCol).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar and step messages (for logs/CI).",
    )
    parser.add_argument(
        "--pycol-no-resume",
        action="store_true",
        help=(
            "Recompute all PyCol metrics from scratch. Default: resume — skip metrics "
            "already saved in --output-csv for this dataset."
        ),
    )
    parser.add_argument(
        "--failure-log",
        default=None,
        metavar="PATH",
        help="Append fatal errors (message + traceback) to this log file.",
    )
    args = parser.parse_args()

    try:
        _run_cli(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        err_line = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {err_line}", file=sys.stderr, flush=True)
        print(tb, file=sys.stderr, flush=True)
        failure_log = Path(args.failure_log) if args.failure_log else None
        if failure_log is not None:
            ds_label = Path(args.ref).name if args.source == "csv" else args.ref
            append_failure_log(
                failure_log,
                "\n".join(
                    [
                        f"========== CLI ERROR {ds_label} ==========",
                        f"ref: {args.ref}",
                        f"source: {args.source}",
                        err_line,
                        tb,
                        "",
                    ]
                ),
            )
        try:
            out_path = Path(args.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            err_result = build_error_result(args, exc)
            merged_df = upsert_result_row(out_path, err_result, key_col=args.upsert_key)
            merged_df.to_csv(out_path, index=False)
            print(f"Wrote error row to: {out_path}", file=sys.stderr, flush=True)
        except Exception as write_exc:
            print(f"Could not write error row to CSV: {write_exc}", file=sys.stderr, flush=True)
        sys.exit(1)


def _run_cli(args: argparse.Namespace) -> None:
    _run_cli_body(args)


def _run_cli_body(args: argparse.Namespace) -> None:
    run_pycol = args.library in ("pycol", "both")
    if run_pycol:
        pycol_arg_chk = (
            args.metrics.strip().lower() if args.library == "pycol" else args.pycol_metrics.strip().lower()
        )
        if pycol_arg_chk == "custom":
            if not args.pycol_custom_metrics or not str(args.pycol_custom_metrics).strip():
                raise ValueError(
                    "When PyCol metrics are 'custom', pass --pycol-custom-metrics with a comma-separated list "
                    "(e.g. --pycol-custom-metrics N1,N3,F1,F1v)."
                )

    show_progress = not args.no_progress

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
    if args.source == "csv":
        result["dataset_file"] = Path(args.ref).name
    if args.upsert_key == "dataset_file" and "dataset_file" not in result:
        result["dataset_file"] = dataset_name if str(dataset_name).endswith(".csv") else f"{dataset_name}.csv"
    if int(args.complexity_max_rows) > 0:
        result["complexity_max_rows"] = int(args.complexity_max_rows)
    if cmeta.get("complexity_subsampled"):
        result["complexity_subsampled"] = True
        result["n_rows_complexity_input"] = int(cmeta["n_rows_complexity_input"])
        result["n_rows_complexity_used"] = int(cmeta["n_rows_complexity_used"])

    n_jobs = int(max(1, args.n_jobs))
    matrix_dtype = np.dtype(args.pycol_matrix_dtype)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        result["pycol_matrix_dtype"] = str(matrix_dtype)
        result["pycol_matrix_storage"] = "ram"
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

        existing_row: dict[str, Any] = {}
        if not args.pycol_no_resume:
            existing_row = load_dataset_result_row(
                out_path, args.upsert_key, str(result[args.upsert_key])
            )
            for k, v in existing_row.items():
                if k.startswith("pycol_") and _stored_value_is_present(v):
                    result[k] = v

        lookup_row = {**existing_row, **result}
        already_done = [m for m in pycol_metrics_run if pycol_metric_is_done(lookup_row, m)]
        pending_metrics = [m for m in pycol_metrics_run if m not in already_done]

        if already_done and not args.pycol_no_resume and show_progress:
            step(
                f"      PyCol resume: {len(already_done)} already saved, "
                f"{len(pending_metrics)} remaining"
            )
            preview = ", ".join(already_done[:10])
            if len(already_done) > 10:
                preview += f", … (+{len(already_done) - 10} more)"
            step(f"        skipped: {preview}")

        checkpoint = make_pycol_checkpoint_saver(
            out_path, result, key_col=args.upsert_key
        )

        n_pycol = int(x_met.shape[0])
        use_sequential_pycol = n_pycol >= PYCOL_SEQUENTIAL_ROW_THRESHOLD and not args.pycol_parallel_metrics
        if not pending_metrics:
            step("      PyCol: all metrics already saved in CSV — nothing to compute.")
        elif use_sequential_pycol:
            phase(
                f"PyCol: computing {len(pending_metrics)} metric(s) sequentially "
                f"({len(already_done)} already saved, single Complexity, n={n_pycol}, "
                f"dtype={matrix_dtype}) …"
            )
            prog_cb = make_pycol_progress_callback(
                show_progress, initial_completed=len(already_done)
            )
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
                    matrix_dtype=matrix_dtype,
                    progress_callback=prog_cb,
                    existing_pycol=existing_row if not args.pycol_no_resume else None,
                    on_metric_complete=checkpoint,
                )
            )
            result["pycol_sequential_large_n"] = True
        else:
            phase(
                f"PyCol: computing {len(pending_metrics)} metric(s) in parallel "
                f"({len(already_done)} already saved) …"
            )
            result.update(
                run_parallel_jobs(
                    "pycol",
                    x_met,
                    y_met,
                    pending_metrics,
                    n_jobs,
                    show_progress=show_progress,
                    desc="pycol",
                    pycol_parallel_heom=bool(args.pycol_parallel_heom)
                    and pycol_matrix_mode != "skip",
                    pycol_heom_n_jobs=n_jobs,
                    pycol_matrix_mode=pycol_matrix_mode,
                    pycol_matrix_dtype=matrix_dtype,
                    on_metric_complete=checkpoint,
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

    phase("Merging and writing CSV …")
    merged_df = upsert_result_row(out_path, result, key_col=args.upsert_key)
    merged_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}", file=sys.stderr if show_progress else sys.stdout, flush=True)
    print(f"Rows: {len(merged_df)}", file=sys.stderr if show_progress else sys.stdout, flush=True)
    print(f"Columns: {len(merged_df.columns)}", file=sys.stderr if show_progress else sys.stdout, flush=True)


if __name__ == "__main__":
    main()

