from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from metric_catalog import PYMFE_COMPLEXITY_METRICS

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


def prepare_xy(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    y_raw = df[label_col].copy()
    x_df = df.drop(columns=[label_col]).copy()

    cat_cols = [
        col
        for col in x_df.columns
        if x_df[col].dtype == "object" or str(x_df[col].dtype).startswith("category")
    ]
    if cat_cols:
        x_df = pd.get_dummies(x_df, columns=cat_cols, dtype=float)

    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

    merged = x_df.copy()
    merged["__target__"] = y_raw
    merged = merged.dropna(axis=0).reset_index(drop=True)

    if merged.empty:
        raise ValueError("No rows left after encoding/cleaning. Check missing values and types.")

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


def basic_info_row(df: pd.DataFrame, x: np.ndarray, y: np.ndarray, label_col: str) -> dict[str, Any]:
    counts = np.bincount(y) if y.size else np.array([0], dtype=int)
    return {
        "label_column": label_col,
        "n_rows_original": int(df.shape[0]),
        "n_columns_original": int(df.shape[1]),
        "n_features_after_encoding": int(x.shape[1]),
        "n_rows_used": int(x.shape[0]),
        "n_classes": int(np.unique(y).size),
        "majority_class_fraction": float(counts.max() / y.size) if y.size else np.nan,
    }


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

