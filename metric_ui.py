from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from complexity_core import (
    METRIC_COST_HEURISTIC_CAPTION,
    PYCOL_MATRIX_MODE_LABELS,
    PYCOL_METRIC_PRESETS,
    PYCOL_METRICS_CHEAP_MINIMAL,
    PYCOL_METRICS_NO_DISTANCE,
    PYCOL_PRESET_MATRIX_MODE,
    PYCOL_PRESET_USER_WHY,
    PycolMatrixMode,
    available_metrics_by_library,
    estimate_heom_matrix_ram_gb,
    get_cheap_expensive_pools,
    partition_pycol_metrics,
    resolve_pycol_matrix_mode,
)

PYCOL_PRESET_LABELS: dict[str, str] = {
    "cheap_minimal": "cheap_minimal — no distance table (fast screening)",
    "cheap": "cheap — 24 metrics, skips T1/NSG/ICSV/ONB/DBC (Hive default)",
    "standard": "standard — cheap + ONB + DBC (26 metrics; slow on large n)",
    "expensive_core": "expensive_core — two tables (T1, NSG, ICSV)",
    "expensive": "expensive — two tables (T1, NSG, ICSV)",
    "all": "all / full — full PyCol catalog (29 metrics)",
    "custom": "custom — you choose metrics",
}

PYCOL_PRESET_ORDER: tuple[str, ...] = (
    "cheap_minimal",
    "cheap",
    "standard",
    "expensive_core",
    "expensive",
    "all",
    "custom",
)


def metric_display_name(name: str, library: str | None = None) -> str:
    """User-facing label with ``pycol_`` or ``pymfe_`` prefix (CSV / chart column style)."""
    n = str(name).strip()
    if n.startswith("pycol_") or n.startswith("pymfe_"):
        return n
    lib = (library or "").strip().lower()
    if lib == "pymfe":
        return f"pymfe_{n}"
    if lib == "pycol":
        return f"pycol_{n}"
    if n and n[0].islower():
        return f"pymfe_{n}"
    return f"pycol_{n}"


def format_metrics_display_list(names: list[str], library: str) -> str:
    return ", ".join(metric_display_name(m, library) for m in names)


@dataclass
class MetricSelectionConfig:
    """Metric lists and PyCol run options (aligned with parallel_complexity_cli)."""

    selected_by_library: dict[str, list[str]]
    pycol_matrix_mode: PycolMatrixMode = "skip"
    pycol_skip_distance_matrix: bool = True
    pycol_parallel_heom: bool = False
    pycol_preset: str | None = None


def render_pycol_resource_warnings(
    *,
    n_rows: int,
    metrics: list[str],
    matrix_mode: PycolMatrixMode,
) -> None:
    """Warn about time/RAM before PyCol runs."""
    no_dist, need_dist = partition_pycol_metrics(metrics)
    label = PYCOL_MATRIX_MODE_LABELS.get(matrix_mode, matrix_mode)
    st.info(f"**HEOM tier (from preset/metrics):** {label}")

    if matrix_mode == "skip":
        if need_dist:
            st.warning(
                "Metrics that need pairwise distances will **not** be computed: "
                f"**{', '.join(need_dist)}**. "
                f"Running: **{', '.join(no_dist) or '(none)'}**."
            )
        return

    if need_dist or not no_dist:
        ram_gb = estimate_heom_matrix_ram_gb(n_rows, matrix_mode)
        st.warning(
            f"At **n = {n_rows:,}**, HEOM matrices need about **{ram_gb:.2f} GB RAM** "
            "(before Python overhead). Time scales roughly with **n²**."
        )
        if n_rows > 20_000 and matrix_mode == "both":
            st.error(
                f"**n = {n_rows:,}** is very large for **two** full matrices (~{ram_gb:.0f} GB). "
                "Use **cheap** preset, subsample rows, or drop T1/NSG/ICSV from custom."
            )
        elif n_rows > 20_000 and matrix_mode == "dist":
            st.warning(
                f"**n = {n_rows:,}** with **one** matrix (~{ram_gb:.0f} GB) — feasible on many servers; still slow."
            )
        elif n_rows > 8_000:
            st.warning("Consider **Max rows** subsampling unless you have plenty of RAM.")


def render_pycol_preset_guide() -> None:
    """Collapsible table of preset categories and why."""
    with st.expander("PyCol preset categories — what & why", expanded=False):
        st.markdown(METRIC_COST_HEURISTIC_CAPTION)
        rows = []
        for key in PYCOL_PRESET_ORDER:
            if key == "custom":
                continue
            tier = PYCOL_PRESET_MATRIX_MODE.get(key, "—")
            rows.append(
                {
                    "Preset": f"`{key}`",
                    "Matrices in RAM": tier,
                    "Why": PYCOL_PRESET_USER_WHY.get(key, ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(PYCOL_PRESET_USER_WHY["custom"])


def render_pycol_preset_metrics(*, key_prefix: str) -> tuple[list[str], str]:
    render_pycol_preset_guide()
    try:
        default_index = PYCOL_PRESET_ORDER.index("cheap")
    except ValueError:
        default_index = 0
    preset = st.selectbox(
        "PyCol preset",
        options=list(PYCOL_PRESET_ORDER),
        format_func=lambda k: PYCOL_PRESET_LABELS.get(k, k),
        index=default_index,
        key=f"{key_prefix}_pycol_preset",
        help="Categories match CLI --metrics and batch PYCOL_METRICS_ARG. HEOM storage is chosen automatically.",
    )
    st.caption(PYCOL_PRESET_USER_WHY.get(preset, ""))
    if preset == "custom":
        all_m = available_metrics_by_library("pycol")
        metrics = st.multiselect(
            "Custom PyCol metrics",
            options=all_m,
            default=list(PYCOL_METRICS_CHEAP_MINIMAL),
            format_func=lambda m: metric_display_name(m, "pycol"),
            key=f"{key_prefix}_pycol_custom_metrics",
        )
    else:
        metrics = list(PYCOL_METRIC_PRESETS[preset])
        tier = PYCOL_PRESET_MATRIX_MODE.get(preset, resolve_pycol_matrix_mode(metrics, preset=preset))
        st.caption(
            f"**{len(metrics)}** metrics · automatic HEOM tier: **{tier}** "
            f"({PYCOL_MATRIX_MODE_LABELS.get(tier, tier)})"
        )
        with st.expander("Metric names in this preset", expanded=(len(metrics) <= 12)):
            st.write(format_metrics_display_list(metrics, "pycol"))
    return metrics, preset


def render_pycol_heom_options(
    metrics: list[str],
    *,
    preset: str | None,
    n_rows: int,
    key_prefix: str,
) -> tuple[PycolMatrixMode, bool]:
    """Returns (matrix_mode, parallel_heom). Tier is chosen from preset/metrics."""
    matrix_mode = resolve_pycol_matrix_mode(metrics, preset=preset)
    st.markdown("**PyCol HEOM (automatic from preset)**")
    st.caption(PYCOL_MATRIX_MODE_LABELS.get(matrix_mode, matrix_mode))
    parallel_heom = False
    if matrix_mode != "skip":
        parallel_heom = st.checkbox(
            "Parallel HEOM build",
            value=True,
            key=f"{key_prefix}_pycol_parallel_heom",
            help="Multi-process row workers (pycol_heom.py).",
        )
    render_pycol_resource_warnings(
        n_rows=n_rows,
        metrics=metrics,
        matrix_mode=matrix_mode,
    )
    return matrix_mode, parallel_heom


# PyCol run metadata stored in CSV — not complexity measure values for bar charts.
_METRIC_COLUMN_SKIP = frozenset(
    {
        "pycol_distance_matrix_skipped",
        "pycol_heom_parallel",
        "pycol_sequential_large_n",
        "complexity_subsampled",
    }
)


def infer_comparison_metric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric pycol_* / pymfe_* columns (coerces strings; skips bool/metadata flags)."""
    out: list[str] = []
    for c in df.columns:
        if c in _METRIC_COLUMN_SKIP:
            continue
        if not (str(c).startswith("pycol_") or str(c).startswith("pymfe_")):
            continue
        if str(c).endswith("_skipped") or str(c).endswith("_parallel"):
            continue
        ser = pd.to_numeric(df[c], errors="coerce")
        if ser.notna().any():
            out.append(c)
    return sorted(out)


def disambiguate_dataset_labels(labels: pd.Series) -> pd.Series:
    """Make row labels unique when the same dataset_name appears more than once."""
    counts: dict[str, int] = {}
    out: list[str] = []
    for raw in labels.astype(str):
        counts[raw] = counts.get(raw, 0) + 1
        n = counts[raw]
        out.append(raw if n == 1 else f"{raw} ({n})")
    return pd.Series(out, index=labels.index)


def prepare_wide_df_for_metric_charts(
    wide_df: pd.DataFrame,
    dataset_field: str,
    metric_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Coerce metrics to float, ensure one chart label per row (duplicate names get #2, #3, …).
    Returns (chart-ready wide frame, list of warnings).
    """
    warnings: list[str] = []
    if dataset_field not in wide_df.columns:
        return wide_df.copy(), ["No dataset identifier column in results."]

    cols = [c for c in metric_columns if c in wide_df.columns]
    if not cols:
        return wide_df.copy(), ["No plottable metric columns in selection."]

    chart_df = wide_df[[dataset_field] + cols].copy()
    chart_df[dataset_field] = chart_df[dataset_field].astype(str)
    n_rows = len(chart_df)
    n_unique = int(chart_df[dataset_field].nunique())
    if n_unique < n_rows:
        warnings.append(
            f"**{n_rows}** result rows share only **{n_unique}** unique `{dataset_field}` — "
            "bars were overlapping. Chart labels disambiguated as `name (2)`, `name (3)`, … "
            "Use a **unique display name** when adding each dataset to the comparison list."
        )
        chart_df[dataset_field] = disambiguate_dataset_labels(chart_df[dataset_field])

    for c in cols:
        chart_df[c] = pd.to_numeric(chart_df[c], errors="coerce")

    return chart_df, warnings


def melt_metrics_for_comparison(
    wide_df: pd.DataFrame,
    dataset_field: str,
    metric_columns: list[str],
) -> pd.DataFrame:
    """Wide table (one row per dataset) → long format for per-metric bar charts."""
    chart_df, _ = prepare_wide_df_for_metric_charts(wide_df, dataset_field, metric_columns)
    cols = [c for c in metric_columns if c in chart_df.columns]
    if not cols or dataset_field not in chart_df.columns:
        return pd.DataFrame(columns=[dataset_field, "metric", "value"])
    long_df = chart_df.melt(
        id_vars=[dataset_field],
        value_vars=cols,
        var_name="metric",
        value_name="value",
    )
    return long_df.dropna(subset=["value"])


def render_per_metric_bar_charts(
    long_df: pd.DataFrame,
    *,
    dataset_field: str = "dataset_name",
    metric_field: str = "metric",
    value_field: str = "value",
    facet_columns: int = 2,
    metrics_order: list[str] | None = None,
) -> None:
    """
    One bar chart per metric; each chart compares all datasets side by side.
    Renders separate Altair charts (not a single faceted chart) so Streamlit shows every dataset.
    """
    import altair as alt

    if long_df.empty:
        st.warning("No numeric values available for selected metrics.")
        return

    long_df = long_df.copy()
    long_df[dataset_field] = long_df[dataset_field].astype(str)
    dataset_order = list(dict.fromkeys(long_df[dataset_field].tolist()))
    n_datasets = len(dataset_order)

    if metrics_order:
        metrics = [m for m in metrics_order if m in set(long_df[metric_field])]
    else:
        metrics = list(dict.fromkeys(long_df[metric_field].tolist()))

    if not metrics:
        st.warning("No plottable metrics in the selection.")
        return

    if n_datasets < 2:
        st.warning(
            f"Only **{n_datasets}** dataset in the chart data (`{dataset_order[0] if dataset_order else '—'}`). "
            "Add more datasets and re-run **Compute comparison metrics**, or upload a CSV with multiple rows."
        )

    st.caption(
        f"**{len(metrics)}** metric chart(s), **{n_datasets}** dataset(s) per chart "
        f"({', '.join(f'`{d}`' for d in dataset_order)}). Y-axis is independent per chart."
    )

    ncol = max(1, min(int(facet_columns), len(metrics)))
    x_sort = dataset_order
    use_horizontal = n_datasets >= 4
    panel_height = max(200, min(320, 120 + 32 * n_datasets)) if use_horizontal else max(180, min(260, 100 + 28 * n_datasets))

    for row_start in range(0, len(metrics), ncol):
        cols = st.columns(ncol)
        for col_idx, metric in enumerate(metrics[row_start : row_start + ncol]):
            sub = long_df[long_df[metric_field] == metric].copy()
            present = set(sub[dataset_field].tolist())
            missing = [d for d in dataset_order if d not in present]
            with cols[col_idx]:
                if use_horizontal:
                    enc = {
                        "y": alt.Y(
                            f"{dataset_field}:N",
                            title="Dataset",
                            sort=x_sort,
                            axis=alt.Axis(labelLimit=200),
                        ),
                        "x": alt.X(f"{value_field}:Q", title="Value", scale=alt.Scale(zero=True)),
                    }
                else:
                    enc = {
                        "x": alt.X(
                            f"{dataset_field}:N",
                            title="Dataset",
                            sort=x_sort,
                            axis=alt.Axis(labelAngle=-25 if n_datasets > 3 else 0),
                        ),
                        "y": alt.Y(f"{value_field}:Q", title="Value", scale=alt.Scale(zero=True)),
                    }
                chart = (
                    alt.Chart(sub)
                    .mark_bar()
                    .encode(
                        **enc,
                        color=alt.Color(f"{dataset_field}:N", legend=None, scale=alt.Scale(domain=x_sort)),
                        tooltip=[
                            alt.Tooltip(f"{dataset_field}:N", title="Dataset"),
                            alt.Tooltip(f"{value_field}:Q", title="Value", format=".6g"),
                        ],
                    )
                    .properties(height=panel_height, title=str(metric))
                )
                st.altair_chart(chart, use_container_width=True)
                if missing:
                    st.caption(f"Missing for: {', '.join(f'`{d}`' for d in missing)}")


def render_cost_tier_metrics(library: str, *, key_prefix: str) -> list[str]:
    """
    PyMFE (and legacy): cheap / expensive pools when not using \"all metrics\".
    """
    cheap_pool, exp_pool = get_cheap_expensive_pools(library)
    if not cheap_pool and library == "pymfe":
        st.warning("No metrics in the cheap pool for this library/runtime.")
        return []

    mode = st.radio(
        f"**{library}** — cost pool mode",
        options=["custom_both", "cheap_only", "expensive_only"],
        format_func=lambda x: {
            "custom_both": "Cheap + expensive (customize each pool)",
            "cheap_only": "Cheap pool only",
            "expensive_only": "Expensive pool only",
        }[x],
        key=f"{key_prefix}_{library}_tier_mode",
        horizontal=False,
    )

    if mode == "cheap_only":
        return st.multiselect(
            f"{library}: metrics in **cheap** pool",
            options=cheap_pool,
            default=cheap_pool,
            format_func=lambda m: metric_display_name(m, library),
            key=f"{key_prefix}_{library}_cheap_only",
        )

    if mode == "expensive_only":
        if not exp_pool:
            st.warning("No metrics in the expensive pool for this library/runtime.")
            return []
        return st.multiselect(
            f"{library}: metrics in **expensive** pool",
            options=exp_pool,
            default=exp_pool,
            format_func=lambda m: metric_display_name(m, library),
            key=f"{key_prefix}_{library}_exp_only",
        )

    st.caption("Selections from both pools are **combined** (union) for this run.")
    c = st.multiselect(
        f"{library}: **cheap** pool",
        options=cheap_pool,
        default=cheap_pool,
        format_func=lambda m: metric_display_name(m, library),
        key=f"{key_prefix}_{library}_cheap_pick",
    )
    e = st.multiselect(
        f"{library}: **expensive** pool",
        options=exp_pool,
        default=exp_pool,
        format_func=lambda m: metric_display_name(m, library),
        key=f"{key_prefix}_{library}_exp_pick",
    )
    return sorted(set(c) | set(e))


def render_metric_selection_block(
    selected_libraries: list[str],
    *,
    key_prefix: str,
    use_all_label: str = "Use all metrics",
    n_rows_for_warnings: int = 0,
) -> MetricSelectionConfig:
    """
    Metric selection for Streamlit (Calculator / Comparison).

    PyCol: CLI-aligned presets (cheap_minimal / cheap / standard / expensive_core / …) + automatic HEOM tier.
    PyMFE: all metrics or cheap/expensive pools.
    """
    selected_by_library: dict[str, list[str]] = {}
    pycol_matrix_mode: PycolMatrixMode = "skip"
    pycol_parallel_heom = False
    pycol_preset: str | None = None

    if not selected_libraries:
        return MetricSelectionConfig(selected_by_library={})

    use_all = st.checkbox(use_all_label, value=True, key=f"{key_prefix}_use_all")

    with st.expander("Metric cost tiers (reference)", expanded=False):
        st.markdown(METRIC_COST_HEURISTIC_CAPTION)
        st.caption(
            "Metrics without pairwise distances: "
            + format_metrics_display_list(sorted(PYCOL_METRICS_NO_DISTANCE), "pycol")
        )

    for lib in selected_libraries:
        st.markdown(f"#### {lib.upper()}")
        if lib == "pycol":
            if use_all:
                selected_by_library[lib] = available_metrics_by_library("pycol")
                st.caption(f"All **{len(selected_by_library[lib])}** PyCol metrics selected.")
                pycol_preset = "all"
                if n_rows_for_warnings > 0:
                    st.warning(
                        "**All PyCol metrics** includes many that need an **n×n distance matrix**. "
                        "This is **time- and memory-intensive** on large datasets."
                    )
                pycol_matrix_mode, pycol_parallel_heom = render_pycol_heom_options(
                    selected_by_library[lib],
                    preset=pycol_preset,
                    n_rows=n_rows_for_warnings,
                    key_prefix=f"{key_prefix}_all",
                )
            else:
                metrics, pycol_preset = render_pycol_preset_metrics(key_prefix=key_prefix)
                selected_by_library[lib] = metrics
                pycol_matrix_mode, pycol_parallel_heom = render_pycol_heom_options(
                    metrics,
                    preset=pycol_preset,
                    n_rows=n_rows_for_warnings,
                    key_prefix=key_prefix,
                )
        else:
            if use_all:
                selected_by_library[lib] = available_metrics_by_library(lib)
                st.caption(f"All **{len(selected_by_library[lib])}** metrics selected.")
                if lib == "pymfe":
                    st.warning(
                        "**All PyMFE complexity features** can be slow on large *n* or wide data "
                        "(neighbor / overlap measures)."
                    )
            else:
                selected_by_library[lib] = render_cost_tier_metrics(lib, key_prefix=key_prefix)

    return MetricSelectionConfig(
        selected_by_library=selected_by_library,
        pycol_matrix_mode=pycol_matrix_mode,
        pycol_skip_distance_matrix=(pycol_matrix_mode == "skip"),
        pycol_parallel_heom=pycol_parallel_heom,
        pycol_preset=pycol_preset,
    )
