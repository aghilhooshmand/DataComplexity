"""Load and filter pre-computed complexity summary tables for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from metric_ui import infer_comparison_metric_columns

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPLEXITY_SUMMARY = PROJECT_ROOT / "results" / "datasets_complexity_summary.csv"
FALLBACK_COMPLEXITY_SUMMARY = PROJECT_ROOT / "pmlb_DS" / "datasets_complexity_summary.csv"


def resolve_complexity_summary_path(path: Path | str | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Complexity summary not found: {p}")
    if DEFAULT_COMPLEXITY_SUMMARY.is_file():
        return DEFAULT_COMPLEXITY_SUMMARY
    if FALLBACK_COMPLEXITY_SUMMARY.is_file():
        return FALLBACK_COMPLEXITY_SUMMARY
    raise FileNotFoundError(
        "No complexity summary CSV found. Expected "
        f"`{DEFAULT_COMPLEXITY_SUMMARY}` or `{FALLBACK_COMPLEXITY_SUMMARY}`."
    )


def dataset_display_name(row: pd.Series) -> str:
    for col in ("dataset_file", "dataset_name", "dataset"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "unknown"


def enrich_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Add display label and PyCol metric completeness columns."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out["display_name"] = out.apply(dataset_display_name, axis=1)

    metric_cols = infer_comparison_metric_columns(out)
    if not metric_cols:
        out["metrics_filled"] = 0
        out["metrics_total"] = 0
        out["completeness_pct"] = 0.0
        return out

    numeric = out[metric_cols].apply(pd.to_numeric, errors="coerce")
    out["metrics_filled"] = numeric.notna().sum(axis=1).astype(int)
    out["metrics_total"] = len(metric_cols)
    out["completeness_pct"] = (out["metrics_filled"] / out["metrics_total"] * 100.0).round(1)
    return out


def _label_missing(series: pd.Series) -> pd.Series:
    """Map null/blank values to a stable sidebar filter label."""
    out = series.astype(str).str.strip()
    return out.mask(series.isna() | (out == "") | (out.str.lower() == "nan"), "(missing)")


_METADATA_NUMERIC_PREFERRED: tuple[str, ...] = (
    "n_rows_used",
    "n_rows_original",
    "n_columns_original",
    "n_features_after_encoding",
    "n_features_raw",
    "n_classes",
    "majority_class_fraction",
    "completeness_pct",
    "metrics_filled",
    "metrics_total",
    "n_jobs",
)

_METADATA_COLUMN_LABELS: dict[str, str] = {
    "n_rows_used": "Rows used (after cleaning)",
    "n_rows_original": "Rows (original)",
    "n_columns_original": "Columns (original, incl. label)",
    "n_features_after_encoding": "Features (after encoding)",
    "n_features_raw": "Features (raw)",
    "n_classes": "Number of classes",
    "majority_class_fraction": "Majority class fraction",
    "completeness_pct": "Metric completeness (%)",
    "metrics_filled": "PyCol metrics filled (count)",
    "metrics_total": "PyCol metrics total (count)",
    "n_jobs": "CLI n_jobs",
}


def metadata_column_label(col: str) -> str:
    return _METADATA_COLUMN_LABELS.get(col, col.replace("_", " "))


def infer_metadata_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric dataset profile / run metadata columns (not pycol_* / pymfe_* metrics)."""
    skip = frozenset(
        {
            "display_name",
            "dataset_file",
            "dataset_name",
            "dataset",
            "error",
            "label_column",
            "missing_values",
            "source",
            "parallel_cli_version",
            "pycol_matrix_dtype",
            "pycol_matrix_mode",
            "pycol_matrix_storage",
            "pycol_metrics_preset",
            "pymfe_groups_used",
            "is_binary",
        }
    )
    out: list[str] = []
    for col in _METADATA_NUMERIC_PREFERRED:
        if col not in df.columns or col in skip:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().any():
            out.append(col)
    for col in df.columns:
        if col in skip or col in out:
            continue
        if str(col).startswith(("pycol_", "pymfe_")):
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().sum() >= max(3, len(df) // 10):
            out.append(col)
    return out


def filter_summary(
    df: pd.DataFrame,
    *,
    search: str = "",
    n_rows_range: tuple[float, float] | None = None,
    n_features_range: tuple[float, float] | None = None,
    n_classes: list[int] | None = None,
    sources: list[str] | None = None,
    completeness: str = "all",
    presets: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to an enriched summary frame."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if search.strip():
        q = search.strip().lower()
        mask = out["display_name"].astype(str).str.lower().str.contains(q, na=False)
        if "dataset_name" in out.columns:
            mask |= out["dataset_name"].astype(str).str.lower().str.contains(q, na=False)
        out = out.loc[mask]

    rows_col = "n_rows_used" if "n_rows_used" in out.columns else "n_rows_original"
    if n_rows_range is not None and rows_col in out.columns:
        ser = pd.to_numeric(out[rows_col], errors="coerce")
        lo, hi = n_rows_range
        out = out.loc[ser.between(lo, hi, inclusive="both")]

    feat_col = (
        "n_features_after_encoding"
        if "n_features_after_encoding" in out.columns
        else "n_features_raw"
    )
    if n_features_range is not None and feat_col in out.columns:
        ser = pd.to_numeric(out[feat_col], errors="coerce")
        lo, hi = n_features_range
        out = out.loc[ser.between(lo, hi, inclusive="both")]

    if n_classes and "n_classes" in out.columns:
        ser = pd.to_numeric(out["n_classes"], errors="coerce")
        out = out.loc[ser.isin(n_classes)]

    if sources and "source" in out.columns:
        out = out.loc[_label_missing(out["source"]).isin(sources)]

    if presets and "pycol_metrics_preset" in out.columns:
        out = out.loc[_label_missing(out["pycol_metrics_preset"]).isin(presets)]

    if completeness != "all" and "metrics_filled" in out.columns:
        if completeness == "complete":
            out = out.loc[out["metrics_filled"] >= out["metrics_total"]]
        elif completeness == "partial":
            out = out.loc[(out["metrics_filled"] > 0) & (out["metrics_filled"] < out["metrics_total"])]
        elif completeness == "none":
            out = out.loc[out["metrics_filled"] == 0]
        elif completeness == "any":
            out = out.loc[out["metrics_filled"] > 0]

    return out.reset_index(drop=True)


def summary_kpis(df: pd.DataFrame) -> dict[str, int | float]:
    if df.empty:
        return {
            "datasets": 0,
            "complete": 0,
            "partial": 0,
            "empty": 0,
            "avg_completeness": 0.0,
        }
    complete = int((df["metrics_filled"] >= df["metrics_total"]).sum()) if "metrics_total" in df.columns else 0
    partial = int(
        ((df["metrics_filled"] > 0) & (df["metrics_filled"] < df["metrics_total"])).sum()
    ) if "metrics_total" in df.columns else 0
    empty = int((df["metrics_filled"] == 0).sum()) if "metrics_filled" in df.columns else len(df)
    avg = float(df["completeness_pct"].mean()) if "completeness_pct" in df.columns else 0.0
    return {
        "datasets": len(df),
        "complete": complete,
        "partial": partial,
        "empty": empty,
        "avg_completeness": round(avg, 1),
    }
