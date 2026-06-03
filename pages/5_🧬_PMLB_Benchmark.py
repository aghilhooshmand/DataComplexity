from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from complexity_core import (
    MISSING_VALUE_LABELS,
    MISSING_VALUE_STRATEGIES,
    prepare_xy,
    run_tsne,
)
from metric_ui import (
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    prepare_wide_df_for_metric_charts,
    render_per_metric_bar_charts,
)
from pmlb_io import (
    DEFAULT_OUTPUT_DIR,
    clean_readme_markdown,
    dataset_name_to_file_key,
    load_cached_metadata_yaml,
    load_cached_readme,
    load_complexity_summary,
    load_encoded_dataset,
    load_manifest,
    load_summary_table,
    lookup_complexity_row,
    metadata_dir_for_dataset,
    metadata_yaml_to_html,
    output_paths,
    pycol_metric_columns,
    row_has_values,
    select_benchmark_datasets,
    stored_metrics_table,
    summary_row_for_dataset,
)
from results_export import render_save_results_section


@st.cache_data(show_spinner=False)
def _cached_complexity_summary() -> pd.DataFrame:
    return load_complexity_summary(DEFAULT_OUTPUT_DIR)


def _render_stored_pycol_metrics(
    stored_row: pd.Series | None,
    complexity_summary: pd.DataFrame,
    *,
    dataset_name: str,
    key_suffix: str,
) -> None:
    st.subheader("PyCol complexity metrics")
    pycol_cols = pycol_metric_columns(complexity_summary) if not complexity_summary.empty else []

    if stored_row is None or not row_has_values(stored_row, pycol_cols):
        st.warning(
            f"No PyCol metrics in `datasets_complexity_summary.csv` for "
            f"**{dataset_name_to_file_key(dataset_name)}**."
        )
        return

    metrics_df = stored_metrics_table(stored_row, pycol_cols)
    if metrics_df.empty:
        st.warning("Summary row exists but all PyCol values are empty.")
        return

    pick = st.multiselect(
        "Metrics to show",
        options=metrics_df["column"].tolist(),
        default=metrics_df["column"].tolist(),
        format_func=str,
        key=f"pmlb_metrics_pick_{key_suffix}",
    )
    view = metrics_df[metrics_df["column"].isin(pick)] if pick else metrics_df
    st.dataframe(view[["metric", "value"]], use_container_width=True, hide_index=True)


def _comparison_table_from_summary(
    complexity_summary: pd.DataFrame,
    dataset_names: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in dataset_names:
        stored = lookup_complexity_row(complexity_summary, name)
        if stored is None:
            rows.append({"dataset": name, "dataset_file": dataset_name_to_file_key(name)})
            continue
        row = stored.to_dict()
        row["dataset"] = name
        rows.append(row)
    return pd.DataFrame(rows)


st.set_page_config(page_title="PMLB Benchmark", layout="wide")
st.title("🧬 PMLB benchmark browser")
st.caption(
    f"Browse PMLB classification datasets and **pre-computed PyCol metrics** "
    f"from `{DEFAULT_OUTPUT_DIR.name}/datasets_complexity_summary.csv`."
)

paths = output_paths(DEFAULT_OUTPUT_DIR)
if not paths["root"].is_dir() or not list(paths["root"].glob("*.csv")):
    st.warning(
        f"No datasets in `{paths['root']}`. Run:\n\n"
        "```bash\n"
        "python download_pmlb_datasets.py\n"
        "```"
    )
    st.stop()

complexity_summary = _cached_complexity_summary()
if complexity_summary.empty:
    st.error(
        "No `pmlb_DS/datasets_complexity_summary.csv` found. "
        "Run the batch script to generate PyCol metrics first."
    )
    st.stop()

manifest = load_manifest(paths["root"])
summary = load_summary_table()
pmlb_available = select_benchmark_datasets(summary, max_instances=21000, pmlb_names=None)

if manifest.empty:
    downloaded = sorted(
        p.stem
        for p in paths["root"].glob("*.csv")
        if p.name != "datasets_complexity_summary.csv"
    )
    catalog = pd.DataFrame({"dataset": downloaded})
else:
    catalog = manifest.copy()

dataset_names = sorted(catalog["dataset"].astype(str).unique().tolist())
if not dataset_names:
    st.error("Manifest is empty and no CSV files found.")
    st.stop()

pycol_cols = pycol_metric_columns(complexity_summary)
n_with_metrics = sum(
    1
    for n in dataset_names
    if lookup_complexity_row(complexity_summary, n) is not None
    and row_has_values(lookup_complexity_row(complexity_summary, n), pycol_cols)
)
st.metric(
    "Datasets with PyCol metrics",
    f"{n_with_metrics} / {len(dataset_names)}",
    help="Loaded from `datasets_complexity_summary.csv` (batch run).",
)

tab_browse, tab_compare, tab_catalog = st.tabs(
    ["Dataset detail", "Compare datasets", "Full catalog"]
)

with tab_catalog:
    st.subheader("Benchmark catalog (summary TSV)")
    st.dataframe(pmlb_available, use_container_width=True, hide_index=True)
    if not complexity_summary.empty and "dataset_file" in complexity_summary.columns:
        st.subheader("PyCol metrics in summary file")
        st.dataframe(
            complexity_summary,
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        f"{len(pmlb_available)} classification datasets in the summary TSV; "
        f"{len(dataset_names)} encoded locally."
    )

with tab_browse:
    ds_name = st.selectbox("Dataset", dataset_names, key="pmlb_ds_select")
    meta_dir = metadata_dir_for_dataset(paths["root"], ds_name)
    stored_row = lookup_complexity_row(complexity_summary, ds_name)

    df_enc: pd.DataFrame | None = None
    try:
        df_enc = load_encoded_dataset(paths["root"], ds_name)
    except FileNotFoundError as exc:
        st.error(str(exc))

    meta_yaml = load_cached_metadata_yaml(meta_dir)
    if meta_yaml:
        st.markdown(metadata_yaml_to_html(meta_yaml, dataset_name=ds_name), unsafe_allow_html=True)
    else:
        st.info(f"No `metadata.yaml` cached for **{ds_name}**.")

    if df_enc is not None:
        n_feat = int(df_enc.shape[1] - 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows (encoded)", f"{df_enc.shape[0]:,}")
        m2.metric("Features", f"{n_feat:,}")
        m3.metric("Classes", int(df_enc["target"].nunique()))
        manifest_row = catalog[catalog["dataset"].astype(str) == ds_name]
        if not manifest_row.empty and "imbalance" in manifest_row.columns:
            imb = manifest_row.iloc[0].get("imbalance")
            m4.metric("Class imbalance", f"{float(imb):.4f}" if pd.notna(imb) else "—")
        else:
            summary_info = summary_row_for_dataset(summary, ds_name)
            if summary_info.get("imbalance") is not None:
                m4.metric("Class imbalance", f"{float(summary_info['imbalance']):.4f}")

    readme = load_cached_readme(meta_dir)
    readme_body = clean_readme_markdown(readme) if readme else ""
    if readme_body:
        with st.expander("README notes", expanded=False):
            st.markdown(readme_body)

    if df_enc is not None:
        with st.expander("Data preview (first 20 rows)", expanded=False):
            st.dataframe(df_enc.head(20), use_container_width=True, hide_index=True)

    st.divider()
    _render_stored_pycol_metrics(
        stored_row,
        complexity_summary,
        dataset_name=ds_name,
        key_suffix=ds_name,
    )

    st.divider()
    st.subheader("t-SNE")
    tsne_missing = st.selectbox(
        "Missing values (t-SNE)",
        options=list(MISSING_VALUE_STRATEGIES),
        index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
        format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
        key="pmlb_tsne_missing",
    )
    if df_enc is not None and st.button("Show t-SNE", key="pmlb_tsne_one"):
        try:
            x, y, _ = prepare_xy(df_enc, "target", missing_values=tsne_missing)
            tsne_df = run_tsne(x, y)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(
                tsne_df["tsne_1"],
                tsne_df["tsne_2"],
                c=tsne_df["label_code"],
                cmap="tab10",
                s=12,
                alpha=0.75,
            )
            ax.set_title(f"t-SNE: {ds_name}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"t-SNE failed: {exc}")

with tab_compare:
    picked = st.multiselect(
        "Datasets to compare",
        dataset_names,
        default=dataset_names[: min(3, len(dataset_names))],
        key="pmlb_compare_pick",
    )
    if not picked:
        st.info("Select at least one dataset.")
    else:
        wide = _comparison_table_from_summary(complexity_summary, picked)
        metric_cols = infer_comparison_metric_columns(wide)

        st.subheader("PyCol metrics (from batch summary)")
        if metric_cols:
            plot_metrics = st.multiselect(
                "Metrics to plot",
                options=metric_cols,
                default=metric_cols[: min(8, len(metric_cols))],
                format_func=str,
                key="pmlb_cmp_plot_metrics",
            )
            if plot_metrics:
                _, chart_warnings = prepare_wide_df_for_metric_charts(
                    wide, dataset_field="dataset", metric_columns=plot_metrics
                )
                for msg in chart_warnings:
                    st.warning(msg)
                long_df = melt_metrics_for_comparison(
                    wide, dataset_field="dataset", metric_columns=plot_metrics
                )
                render_per_metric_bar_charts(
                    long_df,
                    dataset_field="dataset",
                    metrics_order=plot_metrics,
                )
        else:
            st.warning("No plottable PyCol columns for the selected datasets.")

        with st.expander("Full comparison table", expanded=True):
            show_cols = ["dataset", "dataset_file"] + metric_cols
            show_cols = [c for c in show_cols if c in wide.columns]
            st.dataframe(wide[show_cols], use_container_width=True, hide_index=True)

        render_save_results_section(
            wide,
            default_filename="pmlb_comparison_metrics.csv",
            key_prefix="pmlb_cmp_save",
            show_preview=False,
        )

        show_tsne_grid = st.checkbox("Show t-SNE grid", key="pmlb_tsne_grid")
        if show_tsne_grid:
            tsne_missing = st.selectbox(
                "Missing values (t-SNE)",
                options=list(MISSING_VALUE_STRATEGIES),
                index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
                format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
                key="pmlb_cmp_tsne_missing",
            )
            n = len(picked)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
            axes_flat = (
                [axes]
                if n == 1
                else (list(axes.flat) if hasattr(axes, "flat") else list(axes))
            )
            for ax, name in zip(axes_flat, picked):
                try:
                    df_i = load_encoded_dataset(paths["root"], name)
                    x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=tsne_missing)
                    tsne_df = run_tsne(x_i, y_i)
                    ax.scatter(
                        tsne_df["tsne_1"],
                        tsne_df["tsne_2"],
                        c=tsne_df["label_code"],
                        cmap="tab10",
                        s=10,
                        alpha=0.7,
                    )
                    ax.set_title(name[:40])
                except Exception as exc:
                    ax.set_title(f"{name} (failed)")
                    ax.text(
                        0.5, 0.5, str(exc)[:60], ha="center", va="center", transform=ax.transAxes
                    )
            for ax in axes_flat[len(picked) :]:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
