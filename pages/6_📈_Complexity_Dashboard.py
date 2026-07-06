from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from metric_ui import (
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    prepare_wide_df_for_metric_charts,
    render_per_metric_bar_charts,
)
from results_export import render_save_results_section
from summary_dashboard import (
    DEFAULT_COMPLEXITY_SUMMARY,
    _label_missing,
    enrich_summary,
    filter_summary,
    infer_metadata_numeric_columns,
    metadata_column_label,
    resolve_complexity_summary_path,
    summary_kpis,
)


@st.cache_data(show_spinner=False)
def _load_summary(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return enrich_summary(pd.read_csv(path_str))


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _plot_histogram(values: pd.Series, *, title: str, xlabel: str) -> None:
    clean = values.dropna()
    if clean.empty:
        st.info(f"No data for {title}.")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(clean, bins=min(30, max(10, len(clean) // 3)), color="#2563eb", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Datasets")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_scatter(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    color_col: str | None,
    title: str,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        st.warning("Selected columns are not in the filtered table.")
        return
    plot_df = df.copy()
    plot_df["_x"] = _numeric_series(plot_df, x_col)
    plot_df["_y"] = _numeric_series(plot_df, y_col)
    plot_df = plot_df.dropna(subset=["_x", "_y"])
    if plot_df.empty:
        st.info("No numeric points to plot for this selection.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if color_col and color_col in plot_df.columns:
        c = _numeric_series(plot_df, color_col)
        sc = ax.scatter(plot_df["_x"], plot_df["_y"], c=c, cmap="viridis", alpha=0.75, s=48)
        fig.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(plot_df["_x"], plot_df["_y"], color="#2563eb", alpha=0.75, s=48)

    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row.get("display_name", ""))[:22],
            (row["_x"], row["_y"]),
            fontsize=7,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, metric_cols: list[str]) -> None:
    if len(metric_cols) < 2:
        st.info("Pick at least two metrics for a correlation heatmap.")
        return
    sub = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    if sub.notna().sum().sum() < 3:
        st.info("Not enough numeric metric values in the filtered selection.")
        return
    corr = sub.corr()
    size = max(5.0, 0.45 * len(metric_cols))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    short = [c.replace("pycol_", "") for c in metric_cols]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Metric correlation (filtered datasets)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="Complexity Dashboard", layout="wide")
st.title("📈 Complexity results dashboard")
st.caption(
    f"Explore pre-computed PyCol metrics from "
    f"`{DEFAULT_COMPLEXITY_SUMMARY.relative_to(Path(__file__).resolve().parents[1])}`."
)

with st.sidebar:
    st.header("Data source")
    use_upload = st.checkbox("Upload a different summary CSV", value=False)
    uploaded = None
    if use_upload:
        uploaded = st.file_uploader("Summary CSV", type=["csv"], key="dash_summary_upload")

    try:
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            summary_path = Path(uploaded.name)
            mtime_ns = 0
        else:
            summary_path = resolve_complexity_summary_path()
            mtime_ns = summary_path.stat().st_mtime_ns
            raw = None
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if uploaded is not None:
        df_all = enrich_summary(raw)
    else:
        df_all = _load_summary(str(summary_path), mtime_ns)

    if df_all.empty:
        st.error("Summary file has no rows.")
        st.stop()

    st.caption(f"Loaded **{len(df_all)}** datasets from `{summary_path.name}`.")

    if not use_upload and st.button("Reload summary", key="dash_reload"):
        _load_summary.clear()
        st.rerun()

    st.divider()
    st.header("Filters")
    search = st.text_input("Search name", key="dash_search")
    rows_col = "n_rows_used" if "n_rows_used" in df_all.columns else "n_rows_original"
    rows_ser = _numeric_series(df_all, rows_col)
    if rows_ser.notna().any():
        rmin, rmax = float(rows_ser.min()), float(rows_ser.max())
        n_rows_range = st.slider(
            "Rows",
            min_value=rmin,
            max_value=rmax,
            value=(rmin, rmax),
            key="dash_rows",
        )
    else:
        n_rows_range = None

    feat_col = (
        "n_features_after_encoding"
        if "n_features_after_encoding" in df_all.columns
        else "n_features_raw"
    )
    feat_ser = _numeric_series(df_all, feat_col)
    if feat_ser.notna().any():
        fmin, fmax = float(feat_ser.min()), float(feat_ser.max())
        n_features_range = st.slider(
            "Features",
            min_value=fmin,
            max_value=fmax,
            value=(fmin, fmax),
            key="dash_feats",
        )
    else:
        n_features_range = None

    n_classes_opts: list[int] = []
    if "n_classes" in df_all.columns:
        n_classes_opts = sorted(_numeric_series(df_all, "n_classes").dropna().astype(int).unique())
    n_classes = (
        st.multiselect("Classes", options=n_classes_opts, default=n_classes_opts, key="dash_classes")
        if n_classes_opts
        else []
    )

    source_opts = (
        sorted(_label_missing(df_all["source"]).unique()) if "source" in df_all.columns else []
    )
    sources = (
        st.multiselect("Source", options=source_opts, default=source_opts, key="dash_source")
        if source_opts
        else []
    )

    preset_opts = (
        sorted(_label_missing(df_all["pycol_metrics_preset"]).unique())
        if "pycol_metrics_preset" in df_all.columns
        else []
    )
    presets = (
        st.multiselect("PyCol preset", options=preset_opts, default=preset_opts, key="dash_preset")
        if preset_opts
        else []
    )

    completeness = st.selectbox(
        "Metric completeness",
        options=[
            ("all", "All datasets"),
            ("complete", "Fully computed"),
            ("partial", "Partially computed"),
            ("none", "No metrics yet"),
            ("any", "At least one metric"),
        ],
        format_func=lambda x: x[1],
        key="dash_complete",
    )[0]

df = filter_summary(
    df_all,
    search=search,
    n_rows_range=n_rows_range,
    n_features_range=n_features_range,
    n_classes=n_classes or None,
    sources=sources or None,
    presets=presets or None,
    completeness=completeness,
)

metric_cols = infer_comparison_metric_columns(df)
metadata_cols = infer_metadata_numeric_columns(df)
kpis = summary_kpis(df)

m0, m1, m2, m3, m4, m5 = st.columns(6)
m0.metric("Total in file", len(df_all), help="All rows loaded from the summary CSV")
m1.metric("Filtered datasets", kpis["datasets"])
m2.metric("Fully complete", kpis["complete"])
m3.metric("Partial", kpis["partial"])
m4.metric("No metrics", kpis["empty"])
m5.metric("Avg completeness %", kpis["avg_completeness"])

tab_overview, tab_table, tab_compare, tab_charts = st.tabs(
    ["Overview", "Table", "Compare", "Charts"]
)

with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        _plot_histogram(df["completeness_pct"], title="Completeness distribution", xlabel="Metrics filled (%)")
    with c2:
        if rows_col in df.columns:
            _plot_histogram(
                _numeric_series(df, rows_col),
                title="Row count distribution",
                xlabel=metadata_column_label(rows_col),
            )

    if metadata_cols:
        meta_hist = st.multiselect(
            "Dataset metadata distributions",
            metadata_cols,
            default=[c for c in (rows_col, feat_col, "n_columns_original", "n_classes") if c in metadata_cols][:4],
            format_func=metadata_column_label,
            key="dash_overview_meta_hist",
        )
        if meta_hist:
            ncol = min(2, len(meta_hist))
            for i in range(0, len(meta_hist), ncol):
                cols = st.columns(ncol)
                for col_widget, col_name in zip(cols, meta_hist[i : i + ncol]):
                    with col_widget:
                        _plot_histogram(
                            _numeric_series(df, col_name),
                            title=metadata_column_label(col_name),
                            xlabel=col_name,
                        )

    c3, c4 = st.columns(2)
    with c3:
        if metric_cols:
            default_y = "pycol_F1" if "pycol_F1" in metric_cols else metric_cols[0]
            y_pick = st.selectbox("Y metric (overview scatter)", metric_cols, index=metric_cols.index(default_y))
            _plot_scatter(
                df,
                x_col=rows_col if rows_col in df.columns else feat_col,
                y_col=y_pick,
                color_col="n_classes" if "n_classes" in df.columns else None,
                title=f"{rows_col} vs {y_pick}",
            )
    with c4:
        if len(metric_cols) >= 2:
            heat_metrics = st.multiselect(
                "Metrics for correlation heatmap",
                metric_cols,
                default=metric_cols[: min(8, len(metric_cols))],
                key="dash_heat_metrics",
            )
            _plot_correlation_heatmap(df, heat_metrics)

    if not df.empty:
        st.subheader("Lowest completeness")
        show_cols = [
            c
            for c in (
                "display_name",
                "completeness_pct",
                "metrics_filled",
                "metrics_total",
                rows_col,
                feat_col,
                "n_classes",
                "pycol_metrics_preset",
                "error",
            )
            if c in df.columns
        ]
        st.dataframe(
            df.sort_values("completeness_pct").head(15)[show_cols],
            use_container_width=True,
            hide_index=True,
        )

with tab_table:
    sort_by = st.selectbox(
        "Sort by",
        options=["display_name", "completeness_pct", rows_col, feat_col, "n_classes"],
        index=1,
        key="dash_sort",
    )
    ascending = st.checkbox("Ascending", value=True, key="dash_sort_asc")
    table_cols = st.multiselect(
        "Columns",
        options=list(df.columns),
        default=[
            c
            for c in (
                "display_name",
                "completeness_pct",
                "metrics_filled",
                "metrics_total",
                rows_col,
                feat_col,
                "n_classes",
                "source",
                "pycol_metrics_preset",
                "error",
            )
            if c in df.columns
        ]
        + metric_cols[:6],
        key="dash_table_cols",
    )
    view = df.sort_values(sort_by, ascending=ascending, na_position="last")
    if table_cols:
        view = view[table_cols]
    st.dataframe(view, use_container_width=True, hide_index=True)
    render_save_results_section(
        df.sort_values(sort_by, ascending=ascending, na_position="last"),
        default_filename="filtered_complexity_summary.csv",
        key_prefix="dash_export",
        show_preview=False,
    )

with tab_compare:
    names = df["display_name"].astype(str).tolist()
    picked = st.multiselect(
        "Datasets",
        names,
        default=names[: min(5, len(names))],
        key="dash_compare_pick",
    )
    if not picked:
        st.info("Select at least one dataset.")
    else:
        wide = df[df["display_name"].isin(picked)].copy()
        wide["dataset"] = wide["display_name"]
        plot_metrics = st.multiselect(
            "Metrics to compare",
            metric_cols,
            default=metric_cols[: min(10, len(metric_cols))],
            key="dash_compare_metrics",
        )
        if plot_metrics:
            _, warnings = prepare_wide_df_for_metric_charts(
                wide, dataset_field="dataset", metric_columns=plot_metrics
            )
            for msg in warnings:
                st.warning(msg)
            long_df = melt_metrics_for_comparison(
                wide, dataset_field="dataset", metric_columns=plot_metrics
            )
            render_per_metric_bar_charts(
                long_df,
                dataset_field="dataset",
                metrics_order=plot_metrics,
            )
        with st.expander("Comparison table", expanded=True):
            show = ["dataset"] + [c for c in plot_metrics if c in wide.columns]
            st.dataframe(wide[show], use_container_width=True, hide_index=True)

with tab_charts:
    c1, c2, c3 = st.columns(3)
    axis_choices = metadata_cols + metric_cols
    x_col = c1.selectbox(
        "X axis",
        axis_choices,
        index=axis_choices.index(rows_col) if rows_col in axis_choices else 0,
        format_func=lambda c: metadata_column_label(c) if c in metadata_cols else c,
        key="dash_x",
    )
    y_default = metric_cols[0] if metric_cols else (feat_col if feat_col in axis_choices else axis_choices[0])
    y_col = c2.selectbox(
        "Y axis",
        axis_choices,
        index=axis_choices.index(y_default) if y_default in axis_choices else 0,
        format_func=lambda c: metadata_column_label(c) if c in metadata_cols else c,
        key="dash_y",
    )
    color_opts = ["(none)"] + metadata_cols + metric_cols
    color_pick = c3.selectbox(
        "Color",
        color_opts,
        format_func=lambda c: "(none)" if c == "(none)" else (
            metadata_column_label(c) if c in metadata_cols else c
        ),
        key="dash_color",
    )
    color_col = None if color_pick == "(none)" else color_pick
    x_title = metadata_column_label(x_col) if x_col in metadata_cols else x_col
    y_title = metadata_column_label(y_col) if y_col in metadata_cols else y_col
    _plot_scatter(df, x_col=x_col, y_col=y_col, color_col=color_col, title=f"{x_title} vs {y_title}")

    st.subheader("Distributions")
    hist_kind = st.radio(
        "Histogram type",
        ["Dataset metadata", "Complexity metrics"],
        horizontal=True,
        key="dash_hist_kind",
    )
    if hist_kind == "Dataset metadata":
        hist_options = metadata_cols
        hist_default = (
            "n_columns_original"
            if "n_columns_original" in hist_options
            else (rows_col if rows_col in hist_options else (hist_options[0] if hist_options else None))
        )
    else:
        hist_options = metric_cols
        hist_default = "pycol_F1" if "pycol_F1" in hist_options else (hist_options[0] if hist_options else None)

    if hist_options and hist_default:
        hist_col = st.selectbox(
            "Column",
            hist_options,
            index=hist_options.index(hist_default),
            format_func=lambda c: metadata_column_label(c) if c in metadata_cols else c,
            key="dash_hist_col",
        )
        hist_title = metadata_column_label(hist_col) if hist_col in metadata_cols else hist_col
        _plot_histogram(
            _numeric_series(df, hist_col),
            title=f"Distribution of {hist_title}",
            xlabel=hist_col,
        )
    else:
        st.info("No plottable columns for this histogram type.")
