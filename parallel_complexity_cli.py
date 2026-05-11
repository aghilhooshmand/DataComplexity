#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from complexity_core import PYCOL_ALL_METRICS, get_all_pymfe_complexity_metrics, prepare_xy


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


def pycol_metric_job(args: tuple[np.ndarray, np.ndarray, str]) -> tuple[str, Any]:
    x, y, metric = args
    from pycol_complexity import complexity as pycol_complexity

    key = f"pycol_{metric}"
    try:
        comp = pycol_complexity.Complexity(
            file_type="array",
            dataset={"X": np.asarray(x, dtype=float), "y": np.asarray(y)},
            distance_func="default",
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
    if metrics_arg.strip().lower() == "all":
        if lib == "pycol":
            return list(PYCOL_ALL_METRICS)
        if lib == "pymfe":
            return list(get_all_pymfe_complexity_metrics())
        raise ValueError("For library=both, provide --pycol-metrics and --pymfe-metrics.")
    return [m.strip() for m in metrics_arg.split(",") if m.strip()]


def run_parallel_jobs(
    library: str,
    x: np.ndarray,
    y: np.ndarray,
    metrics: list[str],
    n_jobs: int,
) -> dict[str, Any]:
    if not metrics:
        return {}
    worker = pycol_metric_job if library == "pycol" else pymfe_metric_job
    inputs = [(x, y, m) for m in metrics]
    with mp.Pool(processes=n_jobs) as pool:
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
        description="Parallel complexity runner for one dataset (CSV/UCI/OpenML)."
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
        help="Metrics for single library mode (all or comma-separated).",
    )
    parser.add_argument(
        "--pycol-metrics",
        default="all",
        help="For library=both: pycol metrics (all or comma-separated).",
    )
    parser.add_argument(
        "--pymfe-metrics",
        default="all",
        help="For library=both: pymfe metrics (all or comma-separated).",
    )
    parser.add_argument("--n-jobs", type=int, default=max(1, (mp.cpu_count() // 2)))
    parser.add_argument("--output-csv", default="parallel_dataset_complexity.csv")
    args = parser.parse_args()

    df, dataset_name = load_dataset(args.source, args.ref)
    if args.label_column not in df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found. Available columns: {list(df.columns)}"
        )

    x, y, _ = prepare_xy(df, label_col=args.label_column)
    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "source": args.source,
        "label_column": args.label_column,
        "n_rows_original": int(df.shape[0]),
        "n_columns_original": int(df.shape[1]),
        "n_rows_used": int(x.shape[0]),
        "n_features_after_encoding": int(x.shape[1]),
        "n_classes": int(np.unique(y).size),
        "n_jobs": int(max(1, args.n_jobs)),
    }

    n_jobs = int(max(1, args.n_jobs))

    if args.library in ("pycol", "both"):
        pycol_metrics = (
            parse_metrics_arg(args.metrics, "pycol")
            if args.library == "pycol"
            else parse_metrics_arg(args.pycol_metrics, "pycol")
        )
        result.update(run_parallel_jobs("pycol", x, y, pycol_metrics, n_jobs))

    if args.library in ("pymfe", "both"):
        pymfe_metrics = (
            parse_metrics_arg(args.metrics, "pymfe")
            if args.library == "pymfe"
            else parse_metrics_arg(args.pymfe_metrics, "pymfe")
        )
        result.update(run_parallel_jobs("pymfe", x, y, pymfe_metrics, n_jobs))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df = upsert_result_row(out_path, result, key_col="dataset_name")
    merged_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Rows: {len(merged_df)}")
    print(f"Columns: {len(merged_df.columns)}")


if __name__ == "__main__":
    main()

