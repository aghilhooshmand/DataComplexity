from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from metric_catalog import PYMFE_COMPLEXITY_METRICS

# ``event`` is a metric name, ``__init__`` / ``__init_dist__``, or ``done:<metric>``.
# ``index`` is 1-based position when starting a metric, or completed count after ``done:``.
# ``total`` is the number of PyCol metrics in this run.
PycolProgressCallback = Callable[[str, int, int], None]


def _invoke_pycol_progress(
    callback: PycolProgressCallback | Callable[[str], None] | None,
    event: str,
    index: int,
    total: int,
) -> None:
    if callback is None:
        return
    try:
        callback(event, index, total)  # type: ignore[misc]
    except TypeError:
        callback(event)  # type: ignore[misc]

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

# PyCol metrics that use only X/y (class stats, boxes, grids) — no pairwise sample distances.
PYCOL_METRICS_NO_DISTANCE: frozenset[str, ...] = frozenset(
    {
        "F1",
        "F2",
        "F3",
        "F4",
        "F1v",
        "input_noise",
        "purity",
    }
)

# Presets for batch runs (names must match pycol-complexity methods).
PYCOL_METRICS_CHEAP_MINIMAL: tuple[str, ...] = tuple(PYCOL_METRICS_NO_DISTANCE)

# Metrics that read unnorm_dist_matrix (T1 hypersphere radii; NSG/ICSV when sphere_count_method="T1").
PYCOL_METRICS_NEED_UNNORM: frozenset[str, ...] = frozenset({"T1", "NSG", "ICSV"})

# Topology metrics using __get_sphere_count — hours on large n; one normalized matrix only.
PYCOL_METRICS_SLOW_TOPOLOGY: frozenset[str, ...] = frozenset({"ONB", "DBC"})

# standard = all PyCol except T1, NSG, ICSV (one matrix max).
PYCOL_METRICS_STANDARD: tuple[str, ...] = tuple(
    m for m in PYCOL_ALL_METRICS if m not in PYCOL_METRICS_NEED_UNNORM
)

# cheap = standard minus slow topology (ONB, DBC) — practical default for large n on Hive.
PYCOL_METRICS_CHEAP: tuple[str, ...] = tuple(
    m for m in PYCOL_METRICS_STANDARD if m not in PYCOL_METRICS_SLOW_TOPOLOGY
)

# Metrics that require both dist_matrix and unnorm_dist_matrix.
PYCOL_METRICS_EXPENSIVE_CORE: tuple[str, ...] = tuple(PYCOL_METRICS_NEED_UNNORM)

_cheap_set = frozenset(PYCOL_METRICS_CHEAP)
PYCOL_METRICS_EXPENSIVE: tuple[str, ...] = tuple(
    m for m in PYCOL_ALL_METRICS if m in PYCOL_METRICS_NEED_UNNORM
)

PYCOL_METRIC_PRESETS: dict[str, tuple[str, ...]] = {
    "cheap_minimal": PYCOL_METRICS_CHEAP_MINIMAL,
    "cheap": PYCOL_METRICS_CHEAP,
    "standard": PYCOL_METRICS_STANDARD,
    "expensive_core": PYCOL_METRICS_EXPENSIVE_CORE,
    "expensive": PYCOL_METRICS_EXPENSIVE,
    "all": tuple(PYCOL_ALL_METRICS),
}

# RAM tier per preset: skip | dist (normalized only) | both (normalized + unnormalized).
PYCOL_PRESET_MATRIX_MODE: dict[str, Literal["skip", "dist", "both"]] = {
    "cheap_minimal": "skip",
    "cheap": "dist",
    "standard": "dist",
    "expensive_core": "both",
    "expensive": "both",
    "all": "both",
}

PycolMatrixMode = Literal["skip", "dist", "both"]

PYCOL_MATRIX_MODE_LABELS: dict[PycolMatrixMode, str] = {
    "skip": "Level A — no distance matrix",
    "dist": "Level B — normalized dist_matrix only (~½ matrix RAM)",
    "both": "Level C — dist_matrix + unnorm_dist_matrix",
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
    "**RAM tiers:** *cheap_minimal* → **no** distance matrix (F1–F4, F1v, input_noise, purity). "
    "*cheap* → fast benchmark set (excludes T1/NSG/ICSV/ONB/DBC). "
    "*standard* → cheap + ONB + DBC (one matrix). "
    "*full* / *all* → every metric (two matrices when T1/NSG/ICSV run)."
)

# Short “why” text for Streamlit, CLI help, and docs (aligned with PYCOL_PRESET_MATRIX_MODE).
PYCOL_PRESET_USER_WHY: dict[str, str] = {
    "cheap_minimal": (
        "Fastest option: feature overlap and purity only (F1–F4, F1v, input_noise, purity). "
        "No row×row distance table — best for screening many datasets or very large n."
    ),
    "cheap": (
        "Hive default: 24 metrics — all practical PyCol measures except T1, NSG, ICSV, ONB, DBC. "
        "One normalized HEOM matrix when needed; finishes in hours on large n."
    ),
    "standard": (
        "Standard benchmark set: cheap plus ONB and DBC (26 metrics). Still skips T1, NSG, ICSV. "
        "ONB/DBC use slow sphere logic — can add many hours on n≈20k."
    ),
    "expensive_core": (
        "Topology metrics T1, NSG, ICSV only. "
        "Why two matrices: they use unnormalized HEOM (hypersphere radii), not just normalized distances."
    ),
    "expensive": (
        "Same as expensive_core (T1, NSG, ICSV). Use when you only need the two-matrix tier without the full catalog."
    ),
    "all": (
        "Full catalog (29 metrics): cheap + standard-only + expensive_core — two matrices when T1/NSG/ICSV run."
    ),
    "full": (
        "Alias for all — every PyCol metric."
    ),
    "custom": (
        "You pick metric names; the app infers RAM: no table (F1…), one table (N2, N4…), or two tables if T1/NSG/ICSV are selected."
    ),
}

PYCOL_PRESETS_CLI_EPILOG = """
PyCol presets (pick one; --pycol-distance-matrix auto is recommended):

  cheap_minimal   No distance matrix. Fast overlap/purity (F1–F4, F1v, input_noise, purity).

  cheap           24 metrics — all except T1, NSG, ICSV, ONB, DBC (one matrix max). Best for large n.

  standard        26 metrics — cheap + ONB + DBC (still skips T1, NSG, ICSV; one matrix).

  expensive_core  T1, NSG, ICSV only — TWO matrices (normalized + unnormalized HEOM).

  expensive       Same metrics as expensive_core.

  all / full      Full catalog (29 metrics; two matrices when T1/NSG/ICSV run).

  custom          Comma-separated list; RAM tier inferred from names.

HEOM tier (--pycol-distance-matrix): auto | skip | dist | both
  auto  — follows preset above (default)
  dist  — force one matrix; both — force two matrices; skip — force none
""".strip()


def partition_pycol_metrics(metrics: list[str]) -> tuple[list[str], list[str]]:
    """Split metrics into (no pairwise distance matrix, needs distance matrix), preserving order."""
    no_dist = [m for m in metrics if m in PYCOL_METRICS_NO_DISTANCE]
    need_dist = [m for m in metrics if m not in PYCOL_METRICS_NO_DISTANCE]
    return no_dist, need_dist


def stored_value_is_present(val: Any) -> bool:
    """True when a complexity summary cell counts as a saved metric value."""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


def split_pycol_metrics_by_resume(
    metrics: list[str],
    existing: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Return (already_saved, still_pending) for the given metric names."""
    if not existing:
        return [], list(metrics)
    done = [m for m in metrics if stored_value_is_present(existing.get(f"pycol_{m}"))]
    pending = [m for m in metrics if m not in done]
    return done, pending


def pycol_metrics_need_distance_matrix(metrics: list[str]) -> bool:
    """True if any selected metric reads PyCol's n×n distance matrix."""
    _, need_dist = partition_pycol_metrics(metrics)
    return bool(need_dist)


def pycol_metrics_need_unnorm_matrix(metrics: list[str]) -> bool:
    """True if any selected metric may read unnorm_dist_matrix."""
    return bool(set(metrics) & PYCOL_METRICS_NEED_UNNORM)


def resolve_pycol_matrix_mode(
    metrics: list[str],
    *,
    preset: str | None = None,
    override: str | None = None,
) -> PycolMatrixMode:
    """
    Choose HEOM RAM tier from preset and/or metric list.

    - **skip** — no n×n matrix (:data:`PYCOL_METRICS_NO_DISTANCE` only).
    - **dist** — ``dist_matrix`` only (``cheap``, most neighbour metrics).
    - **both** — ``dist_matrix`` + ``unnorm_dist_matrix`` (T1, NSG, ICSV).
    """
    if override is not None:
        key = override.strip().lower()
        aliases = {"build": "auto", "normalized": "dist", "norm": "dist", "unnorm": "both"}
        key = aliases.get(key, key)
        if key == "auto":
            override = None
        elif key in ("skip", "dist", "both"):
            if key == "skip":
                return "skip"
            if key == "dist":
                if not pycol_metrics_need_distance_matrix(metrics):
                    return "skip"
                return "dist"
            if not pycol_metrics_need_distance_matrix(metrics):
                return "skip"
            return "both"
        else:
            raise ValueError(
                f"pycol distance-matrix mode must be skip, dist, both, auto, or build; got {override!r}"
            )

    if preset and preset != "custom" and preset in PYCOL_PRESET_MATRIX_MODE:
        mode = PYCOL_PRESET_MATRIX_MODE[preset]
        if mode == "skip":
            return "skip"
        if not pycol_metrics_need_distance_matrix(metrics):
            return "skip"
        if mode == "both" or pycol_metrics_need_unnorm_matrix(metrics):
            return "both"
        return "dist"

    if not pycol_metrics_need_distance_matrix(metrics):
        return "skip"
    if pycol_metrics_need_unnorm_matrix(metrics):
        return "both"
    return "dist"


def resolve_pycol_skip_distance_matrix(mode: str) -> bool:
    """
    Legacy CLI/batch flag: ``skip`` → True; ``build``/``dist``/``both``/``auto`` → False.

    Prefer :func:`resolve_pycol_matrix_mode` for dist vs both.
    """
    key = mode.strip().lower()
    if key == "skip":
        return True
    if key in ("build", "dist", "both", "auto", "normalized", "norm", "unnorm"):
        return False
    raise ValueError(f"pycol distance-matrix mode must be skip, dist, both, auto, or build; got {mode!r}")


def estimate_heom_matrix_ram_gb(n_rows: int, matrix_mode: PycolMatrixMode) -> float:
    """float64 HEOM matrix RAM (one or two n×n arrays)."""
    n = int(n_rows)
    if n <= 0 or matrix_mode == "skip":
        return 0.0
    matrices = 2 if matrix_mode == "both" else 1
    return matrices * (n**2) * 8.0 / (1024**3)


def parse_pycol_metrics_selection(
    metrics_arg: str, *, custom_metrics: str | None = None
) -> tuple[list[str], str | None]:
    """
    Expand a PyCol metrics argument.

    - ``cheap_minimal`` / ``cheap`` / ``standard`` / ``expensive_core`` / ``expensive`` / ``all`` / ``full`` → built-in lists.
    - ``custom`` → requires ``custom_metrics`` (comma-separated base names); preset ``custom``.
    - Otherwise → ``metrics_arg`` treated as a comma-separated custom list (backward compatible); preset ``custom``.
    """
    key = metrics_arg.strip().lower()
    if key in ("full", "middle"):
        # full → all; middle was renamed to standard
        key = "all" if key == "full" else "standard"
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


def subsample_xy_for_complexity(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    If ``max_rows > 0`` and ``n > max_rows``, take a fixed-seed random subset of rows (no replacement)
    for PyCol / PyMFE / pool workers. Complexity values are then **approximate** for exploratory speed.

    PyCol cost is dominated by ~pairwise structure in *n*; subsampling is the practical way to cap work.
    """
    n = int(x.shape[0])
    meta: dict[str, Any] = {
        "complexity_max_rows": int(max_rows) if max_rows > 0 else None,
        "complexity_subsampled": False,
        "n_rows_complexity_input": n,
        "n_rows_complexity_used": n,
    }
    if max_rows <= 0 or n <= max_rows:
        return x, y, meta
    rng = np.random.default_rng(int(random_state))
    pick = int(min(max_rows, n))
    idx = rng.choice(n, size=pick, replace=False)
    idx.sort()
    meta["complexity_subsampled"] = True
    meta["n_rows_complexity_used"] = pick
    return x[idx], y[idx], meta


def _init_pycol_complexity_shell(
    pycol_complexity: Any,
    x_f: np.ndarray,
    y_a: np.ndarray,
    *,
    dist_matrix: np.ndarray,
    unnorm_dist_matrix: np.ndarray,
) -> Any:
    """Attach X, y, class index attrs, and distance matrices to a bare Complexity instance."""
    comp = pycol_complexity.Complexity.__new__(pycol_complexity.Complexity)
    comp.X = np.array(x_f)
    comp.y = np.array(y_a)
    comp.classes = np.unique(comp.y)
    comp.meta = comp.is_categorical(comp.X)
    comp.dist_matrix = dist_matrix
    comp.unnorm_dist_matrix = unnorm_dist_matrix
    comp.class_count = np.zeros(len(comp.classes), dtype=float)
    for i, cls in enumerate(comp.classes):
        comp.class_count[i] = float(len(np.where(comp.y == cls)[0]))
    comp.class_inxs = [np.where(comp.y == cls)[0] for cls in comp.classes]
    comp.sphere_inst_count_T1 = []
    comp.sphere_tuple_ONB = []
    comp.metrics = {"feature": {}, "struct": {}, "instance": {}, "multi": {}}
    return comp


def build_pycol_complexity(
    x: np.ndarray,
    y: np.ndarray,
    *,
    matrix_mode: PycolMatrixMode = "skip",
    skip_distance_matrix: bool | None = None,
    parallel_heom: bool = False,
    heom_n_jobs: int = 1,
    matrix_dtype: np.dtype | type = np.float64,
) -> Any:
    """
    Construct a PyCol ``Complexity`` instance from in-memory arrays.

    ``matrix_mode``: ``skip`` | ``dist`` (normalized matrix only) | ``both``.
    Legacy ``skip_distance_matrix=True`` forces ``skip`` when ``matrix_mode`` not set.

    Uses :mod:`pycol_heom` (vectorized HEOM). ``parallel_heom=True`` enables row workers.
    """
    from pycol_complexity import complexity as pycol_complexity

    x_f = np.asarray(x, dtype=float)
    y_a = np.asarray(y)

    if skip_distance_matrix is True:
        mode: PycolMatrixMode = "skip"
    else:
        mode = matrix_mode

    if mode == "skip":
        return _init_pycol_complexity_shell(
            pycol_complexity,
            x_f,
            y_a,
            dist_matrix=np.zeros((0, 0), dtype=float),
            unnorm_dist_matrix=np.zeros((0, 0), dtype=float),
        )

    from pycol_heom import build_heom_distance_matrices

    shell = pycol_complexity.Complexity.__new__(pycol_complexity.Complexity)
    meta = shell.is_categorical(np.array(x_f))
    n_jobs = max(1, int(heom_n_jobs)) if parallel_heom else 1
    compute_unnorm = mode == "both"
    dist, unnorm = build_heom_distance_matrices(
        x_f,
        meta,
        n_jobs=n_jobs,
        compute_unnorm=compute_unnorm,
        matrix_dtype=matrix_dtype,
    )
    return _init_pycol_complexity_shell(
        pycol_complexity, x_f, y_a, dist_matrix=dist, unnorm_dist_matrix=unnorm
    )


def _evaluate_pycol_metric(
    comp: Any,
    metric: str,
    *,
    on_failure: Callable[[str, BaseException], None] | None = None,
) -> Any:
    if not hasattr(comp, metric):
        return None
    try:
        val = getattr(comp, metric)()
        if isinstance(val, (float, int, np.floating, np.integer)):
            return float(val)
        arr = np.asarray(val, dtype=float)
        return float(np.nanmean(arr)) if arr.size else None
    except Exception as exc:
        logger.warning("PyCol metric %r failed: %s: %s", metric, type(exc).__name__, exc)
        if on_failure is not None:
            on_failure(metric, exc)
        return None


_PYCOL_SHARED_COMP: Any = None


def _init_pycol_shared_metric_worker(comp: Any) -> None:
    """Pool initializer: share one Complexity instance (fork COW on Linux)."""
    global _PYCOL_SHARED_COMP
    _PYCOL_SHARED_COMP = comp


def _run_shared_pycol_metric_worker(metric: str) -> tuple[str, Any]:
    """Evaluate one PyCol metric using the shared Complexity from the pool initializer."""
    comp = _PYCOL_SHARED_COMP
    if comp is None:
        return metric, None
    return metric, _evaluate_pycol_metric(comp, metric)


def _parallel_metric_pool_workers(requested: int, n_tasks: int) -> int:
    import os

    cpus = max(1, os.cpu_count() or 1)
    return max(1, min(int(requested), int(n_tasks), cpus))


def compute_pycol_metrics(
    x: np.ndarray,
    y: np.ndarray,
    selected_metrics: list[str],
    *,
    matrix_mode: PycolMatrixMode | None = None,
    preset: str | None = None,
    skip_distance_matrix: bool | None = None,
    parallel_heom: bool = False,
    heom_n_jobs: int = 1,
    matrix_dtype: np.dtype | type = np.float64,
    progress_callback: PycolProgressCallback | Callable[[str], None] | None = None,
    existing_pycol: dict[str, Any] | None = None,
    on_metric_complete: Callable[[str, Any], None] | None = None,
    on_metric_failure: Callable[[str, BaseException], None] | None = None,
    parallel_metric_workers: int = 0,
) -> dict[str, Any]:
    import os

    # PyCol + numpy/scipy can spawn many BLAS threads; cap to reduce frozen UI / oversubscription.
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

    if matrix_mode is None:
        if skip_distance_matrix is True:
            mode: PycolMatrixMode = "skip"
        elif skip_distance_matrix is False:
            mode = resolve_pycol_matrix_mode(selected_metrics, preset=preset)
        else:
            mode = resolve_pycol_matrix_mode(selected_metrics, preset=preset)
    else:
        mode = matrix_mode

    no_dist_metrics, need_dist_metrics = partition_pycol_metrics(selected_metrics)
    omitted_need_dist = list(need_dist_metrics) if mode == "skip" else []
    if mode == "skip":
        need_dist_metrics = []

    out: dict[str, Any] = {
        "pycol_matrix_mode": mode,
        "pycol_matrix_dtype": str(np.dtype(matrix_dtype)),
        "pycol_matrix_storage": "ram",
    }
    if omitted_need_dist:
        out["pycol_metrics_omitted_need_distance"] = ",".join(omitted_need_dist)

    if omitted_need_dist:
        out["pycol_metrics_omitted_need_distance"] = ",".join(omitted_need_dist)

    def _metric_already_done(metric: str) -> bool:
        col = f"pycol_{metric}"
        if col in out and stored_value_is_present(out[col]):
            return True
        if existing_pycol is None:
            return False
        val = existing_pycol.get(col)
        if stored_value_is_present(val):
            out[col] = val
            return True
        return False

    all_metric_names = no_dist_metrics + need_dist_metrics
    total_metrics = len(all_metric_names)
    metrics_done = sum(1 for m in all_metric_names if _metric_already_done(m))

    no_dist_pending = [m for m in no_dist_metrics if not _metric_already_done(m)]
    need_dist_pending = [m for m in need_dist_metrics if not _metric_already_done(m)]

    def _metric_start(metric: str) -> None:
        _invoke_pycol_progress(progress_callback, metric, metrics_done + 1, total_metrics)

    def _metric_done(metric: str, value: Any) -> None:
        nonlocal metrics_done
        out[f"pycol_{metric}"] = value
        metrics_done += 1
        _invoke_pycol_progress(
            progress_callback, f"done:{metric}", metrics_done, total_metrics
        )
        if on_metric_complete is not None:
            on_metric_complete(metric, value)

    if no_dist_pending:
        _invoke_pycol_progress(progress_callback, "__init__", metrics_done, total_metrics)
        comp_fast = build_pycol_complexity(
            x, y, matrix_mode="skip", parallel_heom=False, heom_n_jobs=heom_n_jobs
        )
        for metric in no_dist_pending:
            _metric_start(metric)
            _metric_done(
                metric,
                _evaluate_pycol_metric(comp_fast, metric, on_failure=on_metric_failure),
            )
    elif no_dist_metrics and not no_dist_pending and not need_dist_pending:
        out["pycol_distance_matrix_skipped"] = True
    elif no_dist_metrics and not need_dist_metrics:
        out["pycol_distance_matrix_skipped"] = True

    if need_dist_pending:
        _invoke_pycol_progress(progress_callback, "__init_dist__", metrics_done, total_metrics)
        comp_dist = build_pycol_complexity(
            x,
            y,
            matrix_mode=mode,
            parallel_heom=parallel_heom,
            heom_n_jobs=heom_n_jobs,
            matrix_dtype=matrix_dtype,
        )
        if parallel_heom:
            out["pycol_heom_parallel"] = True
        if mode == "dist":
            out["pycol_heom_unnorm_skipped"] = True

        use_shared_parallel = int(parallel_metric_workers) > 1 and len(need_dist_pending) > 1
        fork_ok = False
        if use_shared_parallel:
            try:
                import multiprocessing as mp

                mp.get_context("fork")
                fork_ok = True
            except (ValueError, AttributeError):
                fork_ok = False

        if use_shared_parallel and fork_ok:
            import multiprocessing as mp

            n_workers = _parallel_metric_pool_workers(
                parallel_metric_workers, len(need_dist_pending)
            )
            out["pycol_metrics_parallel_shared"] = True
            out["pycol_metric_workers"] = n_workers
            for metric in need_dist_pending:
                _metric_start(metric)
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_pycol_shared_metric_worker,
                initargs=(comp_dist,),
            ) as pool:
                for metric, value in pool.imap_unordered(
                    _run_shared_pycol_metric_worker, need_dist_pending, chunksize=1
                ):
                    _metric_done(metric, value)
        else:
            if use_shared_parallel and not fork_ok:
                out["pycol_metrics_parallel_shared"] = False
            for metric in need_dist_pending:
                _metric_start(metric)
                _metric_done(
                    metric,
                    _evaluate_pycol_metric(comp_dist, metric, on_failure=on_metric_failure),
                )

    return out


def compute_pymfe_metrics(x: np.ndarray, y: np.ndarray, selected_metrics: list[str]) -> dict[str, Any]:
    import os

    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

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
    import os

    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

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

    PyCol: cheap = :data:`PYCOL_METRICS_CHEAP`; expensive = :data:`PYCOL_METRICS_EXPENSIVE`.

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

