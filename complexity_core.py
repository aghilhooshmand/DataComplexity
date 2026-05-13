from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from metric_catalog import PYMFE_COMPLEXITY_METRICS

MISSING_VALUE_STRATEGIES: tuple[str, ...] = (
    "drop_rows",
    "fill_zero",
    "impute_median",
    "impute_mean",
)

MISSING_VALUE_LABELS: dict[str, str] = {
    "drop_rows": "Drop any row that still has missing values after encoding (strict; can remove many rows)",
    "fill_zero": "Fill remaining missing values with 0",
    "impute_median": "Impute with column median; remaining gaps filled with 0 (good default for messy tabular data)",
    "impute_mean": "Impute with column mean; remaining gaps filled with 0",
}

PYCOL_ALL_METRICS: list[str] = [
    "F1",
    "F1v",
    "F2",
    "F3",
    "F4",
    "input_noise",
    "R_value",
    "deg_overlap",
    "N3",
    "SI",
    "N4",
    "kDN",
    "D3_value",
    "CM",
    "N1",
    "T1",
    "Clust",
    "ONB",
    "LSC",
    "DBC",
    "N2",
    "NSG",
    "ICSV",
    "MRCA",
    "C1",
    "C2",
    "purity",
    "neighbourhood_separability",
    "borderline",
]

# Presets for faster batch runs (names must match pycol-complexity methods).
PYCOL_METRICS_CHEAP: tuple[str, ...] = (
    "F1",
    "F2",
    "F3",
    "N2",
    "N3",
    "C1",
    "C2",
)

PYCOL_METRICS_STRONG: tuple[str, ...] = (
    "F1",
    "F1v",
    "F2",
    "F3",
    "F4",
    "N1",
    "N2",
    "N3",
    "N4",
    "T1",
    "LSC",
    "kDN",
    "C1",
    "C2",
    "purity",
    "borderline",
    "neighbourhood_separability",
    "input_noise",
    "R_value",
)

PYCOL_METRIC_PRESETS: dict[str, tuple[str, ...]] = {
    "cheap": PYCOL_METRICS_CHEAP,
    "strong": PYCOL_METRICS_STRONG,
    "all": tuple(PYCOL_ALL_METRICS),
}

# PyMFE complexity: cheap pool = mostly label / light dimensionality stats (see PyMFE docs:
# ft_c1, ft_c2 use y; ft_t2 uses N only; ft_t4 uses PCA ratio). Neighbor / graph / overlap
# measures are grouped as expensive (heuristic; not an official PyMFE cost table).
PYMFE_METRICS_CHEAP: tuple[str, ...] = (
    "c1",
    "c2",
    "t2",
    "t3",
    "t4",
)

METRIC_COST_HEURISTIC_CAPTION = (
    "**Cost tiers (heuristic):** *Cheap* ≈ class-balance and lightweight dimensionality "
    "summaries. *Expensive* ≈ overlap, PCA-heavy, nearest-neighbor, graph, and MST-style "
    "measures (Lorena survey family). Official PyCol / PyMFE documentation does not give a "
    "complete time-complexity ranking per metric—pools are for practical batch/UI guidance only."
)


def parse_pycol_metrics_selection(
    metrics_arg: str, *, custom_metrics: str | None = None
) -> tuple[list[str], str | None]:
    """
    Expand a PyCol metrics argument.

    - ``cheap`` / ``strong`` / ``all`` → built-in lists (``all`` matches ``PYCOL_ALL_METRICS``).
    - ``custom`` → requires ``custom_metrics`` (comma-separated base names); preset ``custom``.
    - Otherwise → ``metrics_arg`` treated as a comma-separated custom list (backward compatible); preset ``custom``.
    """
    key = metrics_arg.strip().lower()
    if key == "custom":
        if not custom_metrics or not str(custom_metrics).strip():
            raise ValueError(
                "PyCol preset 'custom' requires --pycol-custom-metrics, "
                "e.g. --pycol-custom-metrics F1,N1,N3"
            )
        names = [m.strip() for m in str(custom_metrics).split(",") if m.strip()]
        if not names:
            raise ValueError("--pycol-custom-metrics is empty.")
        return names, "custom"
    if key in PYCOL_METRIC_PRESETS:
        return list(PYCOL_METRIC_PRESETS[key]), key
    names = [m.strip() for m in metrics_arg.split(",") if m.strip()]
    if not names:
        raise ValueError(f"Empty or invalid PyCol metrics argument: {metrics_arg!r}")
    return names, "custom"


def _replace_missing_tokens(series: pd.Series) -> pd.Series:
    """UCI/OpenML often use '?' or empty strings for missing categorical/numeric cells."""
    if series.dtype == "object" or str(series.dtype).startswith("string"):
        s = series.astype(str).str.strip()
        s = s.replace({"?": np.nan, "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
        return s
    return series


def prepare_xy(
    df: pd.DataFrame,
    label_col: str,
    *,
    missing_values: str = "impute_median",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    if missing_values not in MISSING_VALUE_STRATEGIES:
        raise ValueError(
            f"missing_values must be one of {MISSING_VALUE_STRATEGIES}, got {missing_values!r}"
        )

    y_raw = df[label_col].copy()
    x_df = df.drop(columns=[label_col]).copy()

    for col in x_df.columns:
        x_df[col] = _replace_missing_tokens(x_df[col])

    cat_cols = [
        col
        for col in x_df.columns
        if x_df[col].dtype == "object" or str(x_df[col].dtype).startswith("category")
    ]
    if cat_cols:
        x_df = pd.get_dummies(x_df, columns=cat_cols, dtype=float, dummy_na=True)

    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

    if missing_values == "fill_zero":
        x_df = x_df.fillna(0.0)
    elif missing_values == "impute_median":
        med = x_df.median(numeric_only=True)
        x_df = x_df.fillna(med).fillna(0.0)
    elif missing_values == "impute_mean":
        mean = x_df.mean(numeric_only=True)
        x_df = x_df.fillna(mean).fillna(0.0)
    # drop_rows: leave NaNs until row filter below

    merged = x_df.copy()
    merged["__target__"] = y_raw
    merged["__target__"] = _replace_missing_tokens(merged["__target__"])
    merged = merged.dropna(subset=["__target__"], axis=0).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No rows left after dropping rows with missing labels. Check label_column and raw data."
        )

    if missing_values == "drop_rows":
        feat_cols = [c for c in merged.columns if c != "__target__"]
        # Columns that are entirely NaN carry no signal but make every row "incomplete" for
        # listwise deletion — drop them first, then drop rows with any remaining NaN.
        if feat_cols:
            nonempty = [c for c in feat_cols if merged[c].notna().any()]
            dropped_all_nan = sorted(set(feat_cols) - set(nonempty))
            feat_cols = nonempty
            if dropped_all_nan:
                merged = merged.drop(columns=dropped_all_nan, errors="ignore")
        if not feat_cols:
            raise ValueError(
                "drop_rows: no usable feature columns remain (all were entirely NaN after encoding). "
                "Use impute_median / impute_mean / fill_zero, or verify the dataset."
            )
        n_before = int(len(merged))
        row_complete = merged[feat_cols].notna().all(axis=1)
        n_complete = int(row_complete.sum())
        merged_before_row_drop = merged.copy()
        merged = merged.dropna(subset=feat_cols, how="any", axis=0).reset_index(drop=True)
        if merged.empty and n_before > 0:
            nan_frac = {
                c: float(1.0 - merged_before_row_drop[c].notna().mean()) for c in feat_cols
            }
            worst = sorted(nan_frac.items(), key=lambda kv: -kv[1])[:8]
            worst_txt = ", ".join(f"{c} ({p:.0%} missing)" for c, p in worst)
            raise ValueError(
                "drop_rows: no row has complete data in all feature columns after encoding. "
                f"Rows with valid label before row-drop: {n_before}; rows with zero missing features: {n_complete}. "
                f"Highest missing-rate features: {worst_txt}. "
                "For UCI Adult, prefer --missing-values impute_median (or impute_mean). "
                "drop_rows only keeps rows with no NaN in any feature column."
            )

    if merged.empty:
        raise ValueError(
            "No rows left after encoding/cleaning. "
            "Try a different missing-value strategy (e.g. impute_median instead of drop_rows), "
            "or check the label column and raw data."
        )

    x = merged.drop(columns=["__target__"]).to_numpy(dtype=np.float32)
    y_codes, _ = pd.factorize(merged["__target__"], sort=True)
    y = y_codes.astype(int)
    return x, y, merged


def compute_pycol_metrics(x: np.ndarray, y: np.ndarray, selected_metrics: list[str]) -> dict[str, Any]:
    from pycol_complexity import complexity as pycol_complexity

    comp = pycol_complexity.Complexity(
        file_type="array",
        dataset={"X": np.asarray(x, dtype=float), "y": np.asarray(y)},
        distance_func="default",
    )

    out: dict[str, Any] = {}
    for metric in selected_metrics:
        if not hasattr(comp, metric):
            out[f"pycol_{metric}"] = None
            continue
        try:
            val = getattr(comp, metric)()
            if isinstance(val, (float, int, np.floating, np.integer)):
                out[f"pycol_{metric}"] = float(val)
            else:
                arr = np.asarray(val, dtype=float)
                out[f"pycol_{metric}"] = float(np.nanmean(arr)) if arr.size else None
        except Exception:
            out[f"pycol_{metric}"] = None
    return out


def compute_pymfe_metrics(x: np.ndarray, y: np.ndarray, selected_metrics: list[str]) -> dict[str, Any]:
    from pymfe.mfe import MFE

    use_features = selected_metrics if selected_metrics else None
    mfe = MFE(
        groups=["complexity"],
        features=use_features,
        summary=("mean",),
        random_state=0,
    )
    mfe.fit(X=x, y=y)
    names, vals = mfe.extract()

    out: dict[str, Any] = {}
    for name, val in zip(names, vals):
        key = f"pymfe_{name}"
        if isinstance(val, (float, int, np.floating, np.integer)):
            out[key] = float(val)
        else:
            arr = np.asarray(val, dtype=float)
            out[key] = float(np.nanmean(arr)) if arr.size else None
    return out


def run_tsne(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    if x.shape[0] < 3:
        raise ValueError("Need at least 3 rows for t-SNE.")
    emb = TSNE(n_components=2, random_state=42, init="pca").fit_transform(x)
    return pd.DataFrame({"tsne_1": emb[:, 0], "tsne_2": emb[:, 1], "label_code": y})


def basic_info_row(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    label_col: str,
    *,
    missing_values: str | None = None,
) -> dict[str, Any]:
    counts = np.bincount(y) if y.size else np.array([0], dtype=int)
    out: dict[str, Any] = {
        "label_column": label_col,
        "n_rows_original": int(df.shape[0]),
        "n_columns_original": int(df.shape[1]),
        "n_features_after_encoding": int(x.shape[1]),
        "n_rows_used": int(x.shape[0]),
        "n_classes": int(np.unique(y).size),
        "majority_class_fraction": float(counts.max() / y.size) if y.size else np.nan,
    }
    if missing_values is not None:
        out["missing_values"] = missing_values
    return out


def available_metrics_by_library(library: str) -> list[str]:
    if library == "pycol":
        return sorted(PYCOL_ALL_METRICS)
    return sorted(get_all_pymfe_complexity_metrics())


def get_all_pymfe_complexity_metrics() -> list[str]:
    """
    Return all available base feature names from PyMFE complexity group.
    Falls back to curated catalog keys if runtime inspection is unavailable.
    """
    try:
        from pymfe.complexity import MFEComplexity

        names: list[str] = []
        for attr in dir(MFEComplexity):
            if not attr.startswith("ft_"):
                continue
            base = attr[3:]
            if base and not base.startswith("_"):
                names.append(base)
        if names:
            return sorted(set(names))
    except Exception:
        pass
    return sorted(PYMFE_COMPLEXITY_METRICS.keys())


def get_cheap_expensive_pools(library: str) -> tuple[list[str], list[str]]:
    """
    Partition available metrics for ``library`` into cheap vs expensive pools.

    PyCol: cheap = :data:`PYCOL_METRICS_CHEAP`; expensive = remaining names in
    :data:`PYCOL_ALL_METRICS`.

    PyMFE: cheap = :data:`PYMFE_METRICS_CHEAP` intersected with available features;
    expensive = all other available complexity features.
    """
    all_m = available_metrics_by_library(library)
    if library == "pycol":
        cheap_set = set(PYCOL_METRICS_CHEAP)
        cheap = [m for m in all_m if m in cheap_set]
        expensive = [m for m in all_m if m not in cheap_set]
        return cheap, expensive
    if library == "pymfe":
        cheap_set = set(PYMFE_METRICS_CHEAP)
        cheap = [m for m in all_m if m in cheap_set]
        expensive = [m for m in all_m if m not in cheap_set]
        return cheap, expensive
    return [], []

