"""Synthetic Complexity Fusion (SCF): extract class patterns and fuse into new datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from complexity_core import compute_pycol_metrics, prepare_xy
from complexity_profiler import (
    auto_pick_sources_for_metrics,
    profile_summary,
    pycol_key_to_column,
    stress_source_datasets,
)
from pmlb_io import DEFAULT_OUTPUT_DIR, csv_path_for_dataset, dataset_name_to_file_key

FusionMode = Literal["targeted", "stress"]


@dataclass
class ClassStats:
    mean: np.ndarray
    cov: np.ndarray
    n_samples: int


@dataclass
class DatasetPattern:
    """Per-class Gaussian pattern extracted from a real dataset."""

    source_name: str
    source_file: str
    metric_tag: str
    class_stats: dict[int, ClassStats]
    n_features: int


@dataclass
class SyntheticFusionConfig:
    mode: FusionMode = "targeted"
    target_metrics: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    metric_to_source: dict[str, str] = field(default_factory=dict)
    samples_per_class: int = 500
    coupling_noise: float = 0.05
    label_col: str = "target"
    random_seed: int = 42
    pmlb_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    summary_path: Path | None = None
    missing_values: str = "impute_median"
    max_classes: int = 2
    verify_metrics: list[str] = field(default_factory=list)
    output_name: str = "synthetic_fusion"


@dataclass
class SyntheticFusionResult:
    dataframe: pd.DataFrame
    config: SyntheticFusionConfig
    patterns: list[DatasetPattern]
    metric_sources: dict[str, str]
    verified_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _regularized_cov(x: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    if x.shape[0] < 2:
        var = float(np.var(x)) if x.size else 1.0
        return np.eye(x.shape[1], dtype=float) * max(var, reg)
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    return cov + np.eye(cov.shape[0], dtype=float) * reg


def _select_class_indices(y: np.ndarray, max_classes: int) -> list[int]:
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    chosen = classes[order[: max(2, int(max_classes))]]
    if len(chosen) < 2:
        raise ValueError("Need at least two classes for binary synthesis.")
    return [int(c) for c in chosen[:2]]


def load_dataset_for_fusion(
    dataset_ref: str,
    *,
    pmlb_dir: Path,
    label_col: str = "target",
    missing_values: str = "impute_median",
) -> tuple[np.ndarray, np.ndarray, str]:
    file_key = dataset_name_to_file_key(dataset_ref)
    path = csv_path_for_dataset(pmlb_dir, file_key.removesuffix(".csv"))
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    df = pd.read_csv(path)
    if label_col not in df.columns:
        label_col = "target" if "target" in df.columns else df.columns[-1]
    x, y, _ = prepare_xy(df, label_col, missing_values=missing_values)
    return x, y, file_key


def extract_pattern(
    x: np.ndarray,
    y: np.ndarray,
    *,
    source_name: str,
    source_file: str,
    metric_tag: str = "",
    max_classes: int = 2,
) -> DatasetPattern:
    class_ids = _select_class_indices(y, max_classes)
    stats: dict[int, ClassStats] = {}
    for cls in class_ids:
        mask = y == cls
        xs = x[mask]
        stats[cls] = ClassStats(
            mean=np.mean(xs, axis=0),
            cov=_regularized_cov(xs),
            n_samples=int(xs.shape[0]),
        )
    return DatasetPattern(
        source_name=source_name,
        source_file=source_file,
        metric_tag=metric_tag,
        class_stats=stats,
        n_features=int(x.shape[1]),
    )


def _sample_class(stats: ClassStats, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(stats.mean, stats.cov, size=int(n))


def _align_classes(patterns: list[DatasetPattern]) -> tuple[int, int]:
    """Use first two class keys from the first pattern as canonical 0/1."""
    keys = sorted(patterns[0].class_stats.keys())
    if len(keys) < 2:
        raise ValueError("Pattern must have at least two classes.")
    return int(keys[0]), int(keys[1])


def fuse_feature_blocks(
    patterns: list[DatasetPattern],
    *,
    samples_per_class: int,
    coupling_noise: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Concatenate feature blocks — one block per source pattern.
    Each block samples from that pattern's class-conditional Gaussian.
    """
    if not patterns:
        raise ValueError("At least one pattern is required.")
    rng = np.random.default_rng(int(random_seed))
    cls_a, cls_b = _align_classes(patterns)

    n_blocks = len(patterns)
    min_width = min(p.n_features for p in patterns)
    block_width = max(1, min_width // max(1, n_blocks // 2)) if n_blocks > 2 else max(1, min_width)

    feature_names: list[str] = []
    blocks_a: list[np.ndarray] = []
    blocks_b: list[np.ndarray] = []

    for i, pattern in enumerate(patterns):
        start = (i * block_width) % max(1, pattern.n_features - block_width + 1)
        end = start + block_width
        sl = slice(start, end)

        sa = pattern.class_stats[cls_a]
        sb = pattern.class_stats[cls_b]
        mean_a = sa.mean[sl]
        mean_b = sb.mean[sl]
        cov_a = sa.cov[sl, sl]
        cov_b = sb.cov[sl, sl]

        xa = rng.multivariate_normal(mean_a, cov_a, size=int(samples_per_class))
        xb = rng.multivariate_normal(mean_b, cov_b, size=int(samples_per_class))
        blocks_a.append(xa)
        blocks_b.append(xb)
        for j in range(block_width):
            tag = pattern.metric_tag or pattern.source_name
            feature_names.append(f"{tag}_f{start + j}")

    x0 = np.hstack(blocks_a)
    x1 = np.hstack(blocks_b)
    if coupling_noise > 0:
        x0 += rng.normal(0, coupling_noise, size=x0.shape)
        x1 += rng.normal(0, coupling_noise, size=x1.shape)

    x = np.vstack([x0, x1])
    y = np.array([0] * int(samples_per_class) + [1] * int(samples_per_class), dtype=np.int64)
    return x, y, feature_names


def patterns_to_dataframe(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    label_col: str = "target",
) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=feature_names)
    df[label_col] = y
    return df


def resolve_fusion_sources(config: SyntheticFusionConfig) -> tuple[dict[str, str], list[str]]:
    """Return metric→file map and ordered unique source files."""
    profiled = profile_summary(summary_path=config.summary_path)

    if config.mode == "stress":
        if config.source_files:
            files = [dataset_name_to_file_key(f) for f in config.source_files]
        else:
            files = stress_source_datasets(
                profiled,
                metrics=config.target_metrics or None,
                top_n=max(2, len(config.target_metrics) or 4),
            )
        metric_map = {m: files[i % len(files)] for i, m in enumerate(config.target_metrics)} if config.target_metrics else {}
        return metric_map, files

    metrics = config.target_metrics
    if not metrics:
        raise ValueError("targeted mode requires at least one target metric.")

    if config.metric_to_source:
        manual = {k: v for k, v in config.metric_to_source.items()}
        metric_map = auto_pick_sources_for_metrics(profiled, metrics, manual=manual)
    elif config.source_files and len(config.source_files) == len(metrics):
        metric_map = {
            m: dataset_name_to_file_key(f)
            for m, f in zip(metrics, config.source_files, strict=True)
        }
    else:
        metric_map = auto_pick_sources_for_metrics(profiled, metrics)

    ordered: list[str] = []
    for m in metrics:
        f = metric_map.get(m)
        if f and f not in ordered:
            ordered.append(f)
    return metric_map, ordered


def build_patterns_for_fusion(
    config: SyntheticFusionConfig,
    metric_map: dict[str, str],
    source_files: list[str],
) -> list[DatasetPattern]:
    patterns: list[DatasetPattern] = []
    file_to_metric = {}
    for metric, file_key in metric_map.items():
        file_to_metric.setdefault(file_key, metric)

    for file_key in source_files:
        stem = file_key.removesuffix(".csv")
        metric_tag = file_to_metric.get(
            file_key,
            config.target_metrics[len(patterns)] if len(patterns) < len(config.target_metrics) else stem,
        )
        x, y, resolved = load_dataset_for_fusion(
            stem,
            pmlb_dir=config.pmlb_dir,
            label_col=config.label_col,
            missing_values=config.missing_values,
        )
        patterns.append(
            extract_pattern(
                x,
                y,
                source_name=stem,
                source_file=resolved,
                metric_tag=str(metric_tag),
                max_classes=config.max_classes,
            )
        )
    return patterns


def generate_synthetic_dataset(config: SyntheticFusionConfig) -> SyntheticFusionResult:
    metric_map, source_files = resolve_fusion_sources(config)
    if not source_files:
        raise ValueError(
            "No source datasets resolved. Check complexity summary CSV and metric/dataset selections."
        )

    patterns = build_patterns_for_fusion(config, metric_map, source_files)
    x, y, feat_names = fuse_feature_blocks(
        patterns,
        samples_per_class=int(config.samples_per_class),
        coupling_noise=float(config.coupling_noise),
        random_seed=int(config.random_seed),
    )
    df = patterns_to_dataframe(x, y, feat_names, label_col=config.label_col)

    verified: dict[str, float] = {}
    meta_omitted = ""
    if config.verify_metrics:
        from complexity_core import resolve_pycol_matrix_mode

        mode = resolve_pycol_matrix_mode(config.verify_metrics)
        raw = compute_pycol_metrics(
            x.astype(np.float64),
            y,
            config.verify_metrics,
            matrix_mode=mode,
        )
        for m in config.verify_metrics:
            col = pycol_key_to_column(m)
            val = raw.get(col)
            if val is not None:
                verified[m] = float(val)
        omitted = raw.get("pycol_metrics_omitted_need_distance")
        if omitted:
            meta_omitted = str(omitted)
        else:
            meta_omitted = ""

    meta = {
        "mode": config.mode,
        "n_rows": int(len(df)),
        "n_features": int(x.shape[1]),
        "n_classes": 2,
        "source_files": source_files,
        "target_metrics": list(config.target_metrics),
    }
    if config.verify_metrics:
        meta["verified_metrics"] = verified
        if meta_omitted:
            meta["metrics_omitted_verification"] = meta_omitted

    return SyntheticFusionResult(
        dataframe=df,
        config=config,
        patterns=patterns,
        metric_sources=metric_map,
        verified_metrics=verified,
        metadata=meta,
    )


def save_synthetic_dataset(
    result: SyntheticFusionResult,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(path, index=False)
    return path
