"""Rank datasets by PyCol complexity and assign overlap archetypes (SCF Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metric_catalog import PYCOL_METRICS, MetricDirection
from pmlb_io import dataset_name_to_file_key, pycol_metric_columns
from summary_dashboard import dataset_display_name, resolve_complexity_summary_path

# Archetype tags: (name, pycol column stems, hardness percentile threshold).
ARCHETYPE_RULES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    (
        "feature_overlap",
        ("F1", "F2", "F3", "F4", "input_noise"),
        0.75,
    ),
    (
        "neighbor_ambiguity",
        ("N3", "N4", "kDN", "CM", "borderline", "deg_overlap", "SI"),
        0.75,
    ),
    (
        "boundary_structural",
        ("N1", "N2", "T1", "DBC", "ONB", "LSC", "Clust", "NSG", "ICSV"),
        0.75,
    ),
    (
        "multiresolution",
        ("MRCA", "C1", "C2", "purity", "neighbourhood_separability"),
        0.75,
    ),
    (
        "imbalance_overlap",
        ("R_value", "D3_value"),
        0.75,
    ),
)


def pycol_key_to_column(key: str) -> str:
    k = str(key).strip()
    return k if k.startswith("pycol_") else f"pycol_{k}"


def pycol_column_to_key(column: str) -> str:
    c = str(column).strip()
    return c.removeprefix("pycol_")


def metric_hardness_direction(metric_key: str) -> MetricDirection:
    doc = PYCOL_METRICS.get(metric_key) or PYCOL_METRICS.get(metric_key.upper())
    if doc is None:
        return "higher"
    return doc.direction


def hardness_score(series: pd.Series, column: str) -> pd.Series:
    """Map raw metric values to 'harder is larger' scores in [0, 1] via percentile rank."""
    raw = pd.to_numeric(series[column], errors="coerce")
    key = pycol_column_to_key(column)
    direction = metric_hardness_direction(key)
    rank = raw.rank(pct=True, method="average")
    if direction == "lower":
        return 1.0 - rank
    if direction == "context":
        # Treat as weak hardness signal: distance from median.
        med = raw.median()
        spread = raw.std()
        if pd.isna(spread) or spread == 0:
            return rank.fillna(0.5)
        z = (raw - med).abs() / spread
        return z.rank(pct=True, method="average").fillna(0.5)
    return rank


@dataclass(frozen=True)
class ExemplarPick:
    metric: str
    column: str
    dataset_file: str
    dataset_name: str
    raw_value: float
    hardness_rank: float


def load_complexity_summary_table(path: Path | str | None = None) -> pd.DataFrame:
    summary_path = resolve_complexity_summary_path(path)
    df = pd.read_csv(summary_path)
    if "dataset_file" not in df.columns:
        df["dataset_file"] = df.apply(
            lambda r: dataset_name_to_file_key(dataset_display_name(r)),
            axis=1,
        )
    return df


def add_hardness_ranks(summary: pd.DataFrame, metric_columns: list[str] | None = None) -> pd.DataFrame:
    """Add ``hard_<metric>`` percentile columns (higher = harder)."""
    cols = metric_columns or pycol_metric_columns(summary)
    out = summary.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[f"hard_{pycol_column_to_key(col)}"] = hardness_score(out, col)
    return out


def assign_archetypes(
    summary: pd.DataFrame,
    *,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Add ``archetypes`` (comma-separated tags) per dataset row."""
    ranked = add_hardness_ranks(summary)
    tags: list[str] = []

    for _, row in ranked.iterrows():
        row_tags: list[str] = []
        for archetype, stems, rule_thr in ARCHETYPE_RULES:
            thr = threshold if threshold is not None else rule_thr
            hard_cols = [f"hard_{s}" for s in stems if f"hard_{s}" in ranked.columns]
            if not hard_cols:
                continue
            vals = [row[c] for c in hard_cols if pd.notna(row[c])]
            if vals and float(np.nanmean(vals)) >= thr:
                row_tags.append(archetype)
        tags.append(",".join(row_tags) if row_tags else "moderate")
    ranked["archetypes"] = tags
    return ranked


def profile_summary(
    summary: pd.DataFrame | None = None,
    *,
    summary_path: Path | str | None = None,
    threshold: float = 0.75,
) -> pd.DataFrame:
    """Full Phase-1 table: hardness ranks + archetype tags."""
    base = summary if summary is not None else load_complexity_summary_table(summary_path)
    return assign_archetypes(base, threshold=threshold)


def _row_dataset_file(row: pd.Series) -> str:
    if "dataset_file" in row.index and pd.notna(row["dataset_file"]):
        return str(row["dataset_file"]).strip()
    return dataset_name_to_file_key(dataset_display_name(row))


def top_exemplars_for_metric(
    profiled: pd.DataFrame,
    metric: str,
    *,
    top_k: int = 5,
) -> list[ExemplarPick]:
    col = pycol_key_to_column(metric)
    hard_col = f"hard_{pycol_column_to_key(col)}"
    if col not in profiled.columns and hard_col not in profiled.columns:
        return []

    work = profiled.copy()
    if hard_col not in work.columns:
        work = add_hardness_ranks(work, [col])
        hard_col = f"hard_{pycol_column_to_key(col)}"

    work = work[work[col].notna() & work[hard_col].notna()].copy()
    work = work.sort_values(hard_col, ascending=False)

    picks: list[ExemplarPick] = []
    for _, row in work.head(int(top_k)).iterrows():
        picks.append(
            ExemplarPick(
                metric=pycol_column_to_key(col),
                column=col,
                dataset_file=_row_dataset_file(row),
                dataset_name=dataset_display_name(row),
                raw_value=float(row[col]),
                hardness_rank=float(row[hard_col]),
            )
        )
    return picks


def auto_pick_sources_for_metrics(
    profiled: pd.DataFrame,
    metrics: list[str],
    *,
    manual: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Map each metric → dataset_file (exemplar).
    ``manual`` overrides auto picks: metric → dataset stem or file key.
    """
    manual = manual or {}
    out: dict[str, str] = {}
    for metric in metrics:
        key = pycol_column_to_key(pycol_key_to_column(metric))
        if key in manual and str(manual[key]).strip():
            name = str(manual[key]).strip()
            out[key] = dataset_name_to_file_key(name)
            continue
        picks = top_exemplars_for_metric(profiled, key, top_k=1)
        if picks:
            out[key] = picks[0].dataset_file
    return out


def stress_source_datasets(
    profiled: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    top_n: int = 4,
) -> list[str]:
    """Datasets with highest mean hardness across ``metrics`` (or all hard_ columns)."""
    work = profiled.copy()
    if metrics:
        hard_cols = [f"hard_{pycol_column_to_key(pycol_key_to_column(m))}" for m in metrics]
    else:
        hard_cols = [c for c in work.columns if str(c).startswith("hard_")]
    hard_cols = [c for c in hard_cols if c in work.columns]
    if not hard_cols:
        return []

    work["_stress_score"] = work[hard_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    work = work[work["_stress_score"].notna()].sort_values("_stress_score", ascending=False)
    files: list[str] = []
    for _, row in work.iterrows():
        f = _row_dataset_file(row)
        if f not in files:
            files.append(f)
        if len(files) >= int(top_n):
            break
    return files


def dataset_hardness_rank(
    profiled: pd.DataFrame,
    dataset_ref: str,
    metric: str,
) -> float | None:
    """Hardness percentile in [0, 1] for one dataset on one metric (higher = harder)."""
    file_key = dataset_name_to_file_key(dataset_ref)
    hard_col = f"hard_{pycol_column_to_key(pycol_key_to_column(metric))}"
    if "dataset_file" not in profiled.columns or hard_col not in profiled.columns:
        return None
    row = profiled[profiled["dataset_file"].astype(str) == file_key]
    if row.empty or pd.isna(row.iloc[0][hard_col]):
        return None
    return float(row.iloc[0][hard_col])


def exemplar_table(profiled: pd.DataFrame, metrics: list[str], *, top_k: int = 3) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        picks = top_exemplars_for_metric(profiled, metric, top_k=top_k)
        for i, pick in enumerate(picks, start=1):
            rows.append(
                {
                    "metric": pick.metric,
                    "rank": i,
                    "dataset_file": pick.dataset_file,
                    "dataset_name": pick.dataset_name,
                    "value": pick.raw_value,
                    "hardness_rank": pick.hardness_rank,
                }
            )
    return pd.DataFrame(rows)
