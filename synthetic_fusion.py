"""Anchor dataset augmentation: harder samples in the anchor's own classes using donor hardness patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from complexity_core import compute_pycol_metrics, prepare_xy
from complexity_profiler import (
    auto_pick_sources_for_metrics,
    dataset_hardness_rank,
    profile_summary,
    pycol_column_to_key,
    pycol_key_to_column,
)
from pmlb_io import DEFAULT_OUTPUT_DIR, csv_path_for_dataset, dataset_name_to_file_key


@dataclass
class ClassStats:
    mean: np.ndarray
    cov: np.ndarray
    n_samples: int


@dataclass
class DonorProfile:
    metric: str
    donor_name: str
    donor_file: str
    hardness_rank: float
    overlap_intensity: float


@dataclass
class AnchorAugmentConfig:
    """Make one anchor dataset harder by adding samples in its native classes."""

    anchor_dataset: str
    boost_metrics: list[str] = field(default_factory=list)
    donor_per_metric: dict[str, str] = field(default_factory=dict)
    new_samples_per_class: int = 200
    keep_original: bool = True
    perturbation_strength: float = 0.35
    overlap_noise: float = 0.02
    label_col: str = "target"
    random_seed: int = 42
    pmlb_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    summary_path: Path | None = None
    missing_values: str = "impute_median"
    verify_metrics: list[str] = field(default_factory=list)
    output_name: str = "anchor_augmented"


@dataclass
class AnchorAugmentResult:
    dataframe: pd.DataFrame
    config: AnchorAugmentConfig
    donors: list[DonorProfile]
    donor_map: dict[str, str]
    anchor_metrics: dict[str, float] = field(default_factory=dict)
    augmented_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _regularized_cov(x: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    if x.shape[0] < 2:
        var = float(np.var(x)) if x.size else 1.0
        return np.eye(x.shape[1], dtype=float) * max(var, reg)
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    return cov + np.eye(cov.shape[0], dtype=float) * reg


def load_anchor_dataset(
    dataset_ref: str,
    *,
    pmlb_dir: Path,
    label_col: str = "target",
    missing_values: str = "impute_median",
) -> tuple[np.ndarray, np.ndarray, list[str], list[Any], str]:
    """Return encoded X, y codes, feature names, native class labels, file key."""
    file_key = dataset_name_to_file_key(dataset_ref)
    path = csv_path_for_dataset(pmlb_dir, file_key.removesuffix(".csv"))
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    df = pd.read_csv(path)
    if label_col not in df.columns:
        label_col = "target" if "target" in df.columns else df.columns[-1]
    x, y_codes, merged = prepare_xy(df, label_col, missing_values=missing_values)
    feat_cols = [c for c in merged.columns if c != "__target__"]
    _, native_labels = pd.factorize(merged["__target__"], sort=True)
    return x, y_codes.astype(np.int64), feat_cols, list(native_labels), file_key


def overlap_intensity(x: np.ndarray, y: np.ndarray) -> float:
    """How mixed classes are in feature space (higher ≈ more overlap)."""
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.5
    means = {int(c): np.mean(x[y == c], axis=0) for c in classes}
    intra: list[float] = []
    inter: list[float] = []
    for i in range(len(x)):
        c = int(y[i])
        intra.append(float(np.linalg.norm(x[i] - means[c])))
        enemies = [means[e] for e in classes if int(e) != c]
        inter.append(min(float(np.linalg.norm(x[i] - me)) for me in enemies))
    ratio = float(np.mean(np.array(intra) / (np.array(inter) + 1e-9)))
    return float(np.clip(ratio, 0.0, 1.0))


def _class_stats(x: np.ndarray, y: np.ndarray) -> dict[int, ClassStats]:
    stats: dict[int, ClassStats] = {}
    for cls in np.unique(y):
        c = int(cls)
        xs = x[y == c]
        stats[c] = ClassStats(
            mean=np.mean(xs, axis=0),
            cov=_regularized_cov(xs),
            n_samples=int(xs.shape[0]),
        )
    return stats


def resolve_donors(
    config: AnchorAugmentConfig,
    profiled: pd.DataFrame,
) -> tuple[dict[str, str], list[DonorProfile]]:
    if not config.boost_metrics:
        raise ValueError("Select at least one metric to boost (e.g. C1, N3).")

    donor_map = auto_pick_sources_for_metrics(
        profiled,
        config.boost_metrics,
        manual=config.donor_per_metric or None,
    )

    # Never use anchor as its own donor for the same metric.
    anchor_key = dataset_name_to_file_key(config.anchor_dataset)
    for metric, file_key in list(donor_map.items()):
        if file_key == anchor_key:
            picks = profiled.copy()
            col = pycol_key_to_column(metric)
            hard_col = f"hard_{pycol_column_to_key(col)}"
            if col in picks.columns and hard_col in picks.columns:
                alt = picks[
                    (picks["dataset_file"] != anchor_key)
                    & picks[col].notna()
                    & picks[hard_col].notna()
                ].sort_values(hard_col, ascending=False)
                if not alt.empty:
                    donor_map[metric] = str(alt.iloc[0]["dataset_file"])

    donors: list[DonorProfile] = []
    for metric in config.boost_metrics:
        file_key = donor_map.get(metric)
        if not file_key:
            continue
        stem = file_key.removesuffix(".csv")
        rank = dataset_hardness_rank(profiled, file_key, metric)
        if rank is None:
            rank = 0.5
        try:
            dx, dy, _, _, _ = load_anchor_dataset(
                stem,
                pmlb_dir=config.pmlb_dir,
                label_col=config.label_col,
                missing_values=config.missing_values,
            )
            oi = overlap_intensity(dx, dy)
        except Exception:
            oi = 0.5
        donors.append(
            DonorProfile(
                metric=metric,
                donor_name=stem,
                donor_file=file_key,
                hardness_rank=float(rank),
                overlap_intensity=oi,
            )
        )
    return donor_map, donors


def _combined_alpha(donors: list[DonorProfile], base_strength: float) -> float:
    if not donors:
        return float(base_strength)
    strengths = [0.6 * d.hardness_rank + 0.4 * d.overlap_intensity for d in donors]
    return float(np.clip(base_strength * max(strengths), 0.0, 0.95))


def augment_anchor_samples(
    x: np.ndarray,
    y: np.ndarray,
    *,
    new_samples_per_class: int,
    alpha: float,
    overlap_noise: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate new points in each anchor class by sampling the class distribution
    then pushing toward other-class centroids (overlap injection).
    Labels stay the anchor's class codes.
    """
    rng = np.random.default_rng(int(random_seed))
    stats = _class_stats(x, y)
    classes = sorted(int(c) for c in np.unique(y))
    if len(classes) < 2:
        raise ValueError("Anchor dataset must have at least two classes.")

    means = {c: stats[c].mean for c in classes}
    new_x: list[np.ndarray] = []
    new_y: list[int] = []

    for cls in classes:
        enemies = [c for c in classes if c != cls]
        for _ in range(int(new_samples_per_class)):
            base = rng.multivariate_normal(stats[cls].mean, stats[cls].cov)
            enemy = int(rng.choice(enemies))
            direction = means[enemy] - base
            perturbed = base + alpha * direction
            if overlap_noise > 0:
                perturbed += rng.normal(0.0, overlap_noise, size=perturbed.shape)
            new_x.append(perturbed)
            new_y.append(cls)

    return np.vstack(new_x), np.array(new_y, dtype=np.int64)


def _codes_to_native(y_codes: np.ndarray, native_labels: list[Any]) -> np.ndarray:
    return np.array([native_labels[int(c)] for c in y_codes], dtype=object)


def _build_dataframe(
    x: np.ndarray,
    y_codes: np.ndarray,
    feature_cols: list[str],
    native_labels: list[Any],
    *,
    label_col: str,
) -> pd.DataFrame:
    df = pd.DataFrame(x, columns=feature_cols)
    df[label_col] = _codes_to_native(y_codes, native_labels)
    return df


def _compute_metrics(
    x: np.ndarray,
    y: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    if not metrics:
        return {}
    from complexity_core import resolve_pycol_matrix_mode

    mode = resolve_pycol_matrix_mode(metrics)
    raw = compute_pycol_metrics(x.astype(np.float64), y, metrics, matrix_mode=mode)
    out: dict[str, float] = {}
    for m in metrics:
        col = pycol_key_to_column(m)
        val = raw.get(col)
        if val is not None:
            out[m] = float(val)
    return out


def generate_augmented_dataset(config: AnchorAugmentConfig) -> AnchorAugmentResult:
    if not str(config.anchor_dataset).strip():
        raise ValueError("anchor_dataset is required.")

    profiled = profile_summary(summary_path=config.summary_path)
    donor_map, donors = resolve_donors(config, profiled)

    x, y, feat_cols, native_labels, anchor_file = load_anchor_dataset(
        config.anchor_dataset,
        pmlb_dir=config.pmlb_dir,
        label_col=config.label_col,
        missing_values=config.missing_values,
    )

    alpha = _combined_alpha(donors, float(config.perturbation_strength))
    new_x, new_y = augment_anchor_samples(
        x,
        y,
        new_samples_per_class=int(config.new_samples_per_class),
        alpha=alpha,
        overlap_noise=float(config.overlap_noise),
        random_seed=int(config.random_seed),
    )

    if config.keep_original:
        x_out = np.vstack([x, new_x])
        y_out = np.concatenate([y, new_y])
    else:
        x_out, y_out = new_x, new_y

    df = _build_dataframe(x_out, y_out, feat_cols, native_labels, label_col=config.label_col)

    verify = list(config.verify_metrics) or list(config.boost_metrics)
    anchor_metrics = _compute_metrics(x, y, verify) if verify else {}
    augmented_metrics = _compute_metrics(x_out, y_out, verify) if verify else {}

    meta = {
        "anchor_dataset": config.anchor_dataset,
        "anchor_file": anchor_file,
        "boost_metrics": list(config.boost_metrics),
        "donor_map": donor_map,
        "perturbation_alpha": alpha,
        "n_rows_original": int(len(x)),
        "n_rows_new": int(len(new_x)),
        "n_rows_total": int(len(x_out)),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "native_class_labels": [str(v) for v in native_labels],
        "keep_original": bool(config.keep_original),
    }

    return AnchorAugmentResult(
        dataframe=df,
        config=config,
        donors=donors,
        donor_map=donor_map,
        anchor_metrics=anchor_metrics,
        augmented_metrics=augmented_metrics,
        metadata=meta,
    )


def save_augmented_dataset(result: AnchorAugmentResult, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(path, index=False)
    return path


# Backwards-compatible aliases for imports that still use old names.
SyntheticFusionConfig = AnchorAugmentConfig
SyntheticFusionResult = AnchorAugmentResult
generate_synthetic_dataset = generate_augmented_dataset
save_synthetic_dataset = save_augmented_dataset
