#!/usr/bin/env python3
"""PyCol N3 only — uses vendored pycol_fork with parallel N3 (n_jobs)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from complexity_core import MISSING_VALUE_STRATEGIES, prepare_xy, subsample_xy_for_complexity
from pycol_fork.pycol_complexity.complexity import Complexity

CLI_VERSION = "1.0.0"


def extract_last_int(text: str) -> int | None:
    m = re.findall(r"\d+", str(text))
    return int(m[-1]) if m else None


def load_dataset(source: str, ref: str) -> tuple[pd.DataFrame, str]:
    ref = str(ref).strip()
    s = source.strip().lower()
    if s == "csv":
        p = Path(ref)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return pd.read_csv(p), p.stem
    if s == "uci":
        from ucimlrepo import fetch_ucirepo

        ds_id = extract_last_int(ref)
        if ds_id is None:
            raise ValueError(f"Could not parse UCI id from: {ref}")
        ds = fetch_ucirepo(id=int(ds_id))
        x, y = ds.data.features, ds.data.targets
        if y is None:
            raise ValueError("UCI dataset has no target.")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        df = x.copy()
        df["target"] = y
        return df, f"uci_{ds_id}"
    if s == "openml":
        import openml

        ds_id = extract_last_int(ref)
        if ds_id is None:
            raise ValueError(f"Could not parse OpenML id from: {ref}")
        ds = openml.datasets.get_dataset(int(ds_id))
        x, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        df = x.copy()
        df["target"] = y
        return df, f"openml_{ds_id}"
    raise ValueError("source must be csv, uci, or openml")


def upsert_result_row(output_csv: Path, result: dict[str, Any], key_col: str = "dataset_name") -> pd.DataFrame:
    new_row_df = pd.DataFrame([result])
    if not output_csv.exists():
        return new_row_df
    existing = pd.read_csv(output_csv)
    if existing.empty:
        return new_row_df
    if key_col not in existing.columns:
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
    p = argparse.ArgumentParser(description="PyCol N3 (parallel fork).")
    p.add_argument("--source", required=True, choices=["csv", "uci", "openml"])
    p.add_argument("--ref", required=True)
    p.add_argument("--label-column", default="target")
    p.add_argument("--missing-values", default="impute_median", choices=list(MISSING_VALUE_STRATEGIES))
    p.add_argument("--complexity-max-rows", type=int, default=0)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--imb", action="store_true")
    p.add_argument("--inst-level", action="store_true")
    p.add_argument("--n-jobs", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    p.add_argument(
        "--output-csv",
        default=None,
        help="If set, upsert one row per dataset (column pycol_N3). For batch scripts.",
    )
    p.add_argument("--json", action="store_true", help="Print result JSON to stdout (in addition to CSV if set).")
    p.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")
    args = p.parse_args()

    df, name = load_dataset(args.source, args.ref)
    if args.label_column not in df.columns:
        raise SystemExit(f"Missing label column {args.label_column!r}")
    x, y, _ = prepare_xy(df, label_col=args.label_column, missing_values=args.missing_values)
    x_m, y_m, meta = subsample_xy_for_complexity(x, y, int(args.complexity_max_rows))

    comp = Complexity(
        file_type="array",
        dataset={"X": np.asarray(x_m, dtype=float), "y": np.asarray(y_m)},
        distance_func="default",
    )
    n3 = comp.N3(k=int(args.k), imb=bool(args.imb), inst_level=bool(args.inst_level), n_jobs=int(args.n_jobs))

    result: dict[str, Any] = {
        "dataset_name": name,
        "n3_cli_version": CLI_VERSION,
        "source": args.source,
        "label_column": args.label_column,
        "missing_values": args.missing_values,
        "k": int(args.k),
        "n_rows_original": int(df.shape[0]),
        "n_rows_used": int(x.shape[0]),
        "n_rows_n3": int(x_m.shape[0]),
        "n_features_after_encoding": int(x_m.shape[1]),
        "n_classes": int(np.unique(y_m).size),
        "n_jobs": int(args.n_jobs),
    }
    if int(args.complexity_max_rows) > 0:
        result["complexity_max_rows"] = int(args.complexity_max_rows)
    if meta.get("complexity_subsampled"):
        result["complexity_subsampled"] = True
        result["n_rows_complexity_input"] = int(meta["n_rows_complexity_input"])
        result["n_rows_complexity_used"] = int(meta["n_rows_complexity_used"])

    if args.inst_level:
        result["pycol_n3_inst_hardness"] = [float(v) for v in np.asarray(n3).ravel()]
    elif args.imb:
        for i, c in enumerate(comp.classes):
            result[f"pycol_N3_class_{int(c)}"] = float(np.asarray(n3).ravel()[i])
    else:
        result["pycol_N3"] = float(n3)

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged = upsert_result_row(out_path, result, key_col="dataset_name")
        merged.to_csv(out_path, index=False)
        print(f"Saved: {out_path}", file=sys.stderr, flush=True)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.output_csv:
        print(result)


if __name__ == "__main__":
    main()
