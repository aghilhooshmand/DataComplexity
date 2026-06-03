from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from complexity_core import (
    MISSING_VALUE_LABELS,
    MISSING_VALUE_STRATEGIES,
    basic_info_row,
    compute_pycol_metrics,
    compute_pymfe_metrics,
    prepare_xy,
    run_tsne,
    subsample_xy_for_complexity,
)
from metric_ui import (
    format_metrics_display_list,
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    prepare_wide_df_for_metric_charts,
    render_metric_selection_block,
    render_per_metric_bar_charts,
)
from pmlb_io import (
    DEFAULT_OUTPUT_DIR,
    clean_readme_markdown,
    dataset_name_to_file_key,
    datasets_with_pycol,
    load_cached_metadata_yaml,
    load_cached_readme,
    load_complexity_summary,
    load_encoded_dataset,
    load_manifest,
    load_summary_table,
    lookup_complexity_row,
    merge_metric_row,
    metadata_dir_for_dataset,
    metadata_yaml_to_html,
    missing_pycol_keys,
    output_paths,
    pycol_metric_columns,
    row_has_values,
    save_complexity_result_row,
    select_benchmark_datasets,
    stored_metrics_table,
    summary_row_for_dataset,
)
from results_export import render_save_results_section


@st.cache_data(show_spinner=False)
def _cached_complexity_summary() -> pd.DataFrame:
    return load_complexity_summary(DEFAULT_OUTPUT_DIR)


def _render_pycol_metrics_panel(
    *,
    dataset_name: str,
    df_enc: pd.DataFrame | None,
    stored_row: pd.Series | None,
    complexity_summary: pd.DataFrame,
    key_prefix: str,
) -> None:
    st.subheader("PyCol complexity metrics")
    pycol_cols = pycol_metric_columns(complexity_summary) if not complexity_summary.empty else []
    has_stored = stored_row is not None and row_has_values(stored_row, pycol_cols)

    if has_stored:
        st.success(
            f"Pre-computed metrics found in `datasets_complexity_summary.csv` "
            f"(`{dataset_name_to_file_key(dataset_name)}`)."
        )
    else:
        st.info("No pre-computed PyCol metrics for this dataset yet. Compute below.")

    display_row: dict[str, Any] | None = None
    session_key = f"pmlb_metrics_{key_prefix}_{dataset_name}"

    if has_stored and stored_row is not None:
        all_stored = stored_metrics_table(stored_row, pycol_cols)
        if not all_stored.empty:
            with st.expander("View stored metrics", expanded=True):
                pick = st.multiselect(
                    "Metrics to show",
                    options=all_stored["column"].tolist(),
                    default=all_stored["column"].tolist(),
                    format_func=str,
                    key=f"{key_prefix}_stored_pick_{dataset_name}",
                )
                view = all_stored[all_stored["column"].isin(pick)] if pick else all_stored
                st.dataframe(
                    view[["metric", "value"]],
                    use_container_width=True,
                    hide_index=True,
                )
        display_row = stored_row.to_dict()

    if session_key in st.session_state:
        display_row = merge_metric_row(display_row, st.session_state[session_key])

    st.markdown("#### Compute additional metrics")
    missing_values = st.selectbox(
        "Missing values (for new computation)",
        options=list(MISSING_VALUE_STRATEGIES),
        index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
        format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
        key=f"{key_prefix}_missing_{dataset_name}",
    )
    complexity_max_rows = st.number_input(
        "Max rows (0 = all)",
        min_value=0,
        max_value=500_000,
        value=0,
        step=500,
        key=f"{key_prefix}_maxrows_{dataset_name}",
    )
    selected_libraries = st.multiselect(
        "Libraries",
        options=["pycol", "pymfe"],
        default=["pycol"],
        key=f"{key_prefix}_libs_{dataset_name}",
    )
    metric_config = render_metric_selection_block(
        selected_libraries,
        key_prefix=f"{key_prefix}_{dataset_name}",
        use_all_label="Use all metrics",
        n_rows_for_warnings=int(df_enc.shape[0]) if df_enc is not None else 0,
    )
    recompute_existing = st.checkbox(
        "Recompute metrics that are already stored",
        value=False,
        key=f"{key_prefix}_recompute_{dataset_name}",
    )
    save_to_csv = st.checkbox(
        "Save results to `pmlb_DS/datasets_complexity_summary.csv`",
        value=False,
        key=f"{key_prefix}_save_{dataset_name}",
    )

    requested_pycol = metric_config.selected_by_library.get("pycol", [])
    if requested_pycol and has_stored and stored_row is not None and not recompute_existing:
        missing_keys = missing_pycol_keys(stored_row, requested_pycol)
        if missing_keys:
            st.caption(
                "Will compute **"
                + ", ".join(missing_keys)
                + "** (already stored metrics are skipped)."
            )
        else:
            st.caption("All selected PyCol metrics are already stored. Enable recompute to refresh.")

    col_run, col_clear = st.columns(2)
    with col_run:
        run = st.button("Compute selected metrics", key=f"{key_prefix}_compute_{dataset_name}")
    with col_clear:
        if st.button("Clear session results", key=f"{key_prefix}_clear_{dataset_name}"):
            st.session_state.pop(session_key, None)
            st.rerun()

    if run:
        if df_enc is None:
            st.error("Load the dataset CSV first.")
        elif not selected_libraries:
            st.error("Select at least one library.")
        else:
            try:
                x, y, _ = prepare_xy(df_enc, "target", missing_values=missing_values)
                x, y, sub_meta = subsample_xy_for_complexity(
                    x, y, int(complexity_max_rows), random_state=0
                )
                computed: dict[str, Any] = {
                    "dataset": dataset_name,
                    "dataset_file": dataset_name_to_file_key(dataset_name),
                    **basic_info_row(df_enc, x, y, "target", missing_values=missing_values),
                    **sub_meta,
                }
                if "pycol" in selected_libraries:
                    to_run = list(requested_pycol)
                    if has_stored and stored_row is not None and not recompute_existing:
                        missing_keys = missing_pycol_keys(stored_row, to_run)
                        to_run = [k.replace("pycol_", "") for k in missing_keys]
                    if to_run:
                        computed.update(
                            compute_pycol_metrics(
                                x,
                                y,
                                to_run,
                                preset=metric_config.pycol_preset,
                            )
                        )
                    elif not recompute_existing:
                        st.warning("Nothing new to compute for PyCol with current selection.")
                if "pymfe" in selected_libraries:
                    computed.update(
                        compute_pymfe_metrics(
                            x, y, metric_config.selected_by_library.get("pymfe", [])
                        )
                    )
                if len(computed) > 5:
                    st.session_state[session_key] = computed
                    display_row = merge_metric_row(display_row, computed)
                    if save_to_csv:
                        path = save_complexity_result_row(computed)
                        st.success(f"Saved to `{path}`")
                        _cached_complexity_summary.clear()
                    else:
                        st.success("Metrics computed (session only). Enable save to update the CSV.")
            except Exception as exc:
                st.error(f"Computation failed: {exc}")

    if display_row:
        live_cols = [
            c
            for c in pycol_metric_columns(pd.DataFrame([display_row]))
            if c in display_row and pd.notna(display_row.get(c))
        ]
        if live_cols:
            st.markdown("#### Current view (stored + session)")
            live_df = stored_metrics_table(pd.Series(display_row), live_cols)
            st.dataframe(live_df[["metric", "value"]], use_container_width=True, hide_index=True)


st.set_page_config(page_title="PMLB Benchmark", layout="wide")
st.title("🧬 PMLB benchmark browser")
st.caption(
    f"Browse locally downloaded PMLB classification datasets "
    f"(< 21k rows, encoded CSVs in `{DEFAULT_OUTPUT_DIR.name}/`)."
)

paths = output_paths(DEFAULT_OUTPUT_DIR)
if not paths["root"].is_dir() or not list(paths["root"].glob("*.csv")):
    st.warning(
        f"No datasets in `{paths['root']}`. Run:\n\n"
        "```bash\n"
        "python download_pmlb_datasets.py\n"
        "```\n\n"
        "Requires `pip install pmlb pyyaml`."
    )
    st.stop()

complexity_summary = _cached_complexity_summary()
manifest = load_manifest(paths["root"])
summary = load_summary_table()
pmlb_available = select_benchmark_datasets(summary, max_instances=21000, pmlb_names=None)

if manifest.empty:
    downloaded = sorted(p.stem for p in paths["root"].glob("*.csv"))
    catalog = pd.DataFrame({"dataset": downloaded})
else:
    catalog = manifest.copy()

dataset_names = sorted(catalog["dataset"].astype(str).unique().tolist())
if not dataset_names:
    st.error("Manifest is empty and no CSV files found.")
    st.stop()

pycol_done = datasets_with_pycol(complexity_summary)
local_files = {dataset_name_to_file_key(n) for n in dataset_names}
n_precomputed = len(pycol_done & local_files)
st.metric(
    "PyCol pre-computed (local PMLB CSVs)",
    f"{n_precomputed} / {len(local_files)}",
    help="Rows in `pmlb_DS/datasets_complexity_summary.csv` with at least one non-null PyCol metric.",
)

tab_browse, tab_compare, tab_catalog = st.tabs(
    ["Dataset detail", "Compare datasets", "Full catalog"]
)

with tab_catalog:
    st.subheader("Benchmark catalog (summary TSV)")
    st.dataframe(
        pmlb_available,
        use_container_width=True,
        hide_index=True,
    )
    if not complexity_summary.empty and "dataset_file" in complexity_summary.columns:
        st.subheader("PyCol coverage (batch summary)")
        cov_rows = []
        for name in dataset_names:
            fk = dataset_name_to_file_key(name)
            cov_rows.append(
                {
                    "dataset": name,
                    "dataset_file": fk,
                    "pycol_precomputed": fk in pycol_done,
                }
            )
        cov_df = pd.DataFrame(cov_rows)
        st.dataframe(
            cov_df.sort_values(["pycol_precomputed", "dataset"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        f"{len(pmlb_available)} classification datasets with n_instances < 21,000 in the summary file; "
        f"{len(dataset_names)} encoded locally."
    )
    if paths["status"].is_file():
        with st.expander("Last download run"):
            st.json(paths["status"].read_text(encoding="utf-8"))

with tab_browse:
    ds_name = st.selectbox("Dataset", dataset_names, key="pmlb_ds_select")
    meta_dir = metadata_dir_for_dataset(paths["root"], ds_name)
    meta_yaml = load_cached_metadata_yaml(meta_dir)
    readme = load_cached_readme(meta_dir)
    stored_row = lookup_complexity_row(complexity_summary, ds_name)

    df_enc: pd.DataFrame | None = None
    try:
        df_enc = load_encoded_dataset(paths["root"], ds_name)
    except FileNotFoundError as exc:
        st.error(str(exc))

    if meta_yaml:
        st.markdown(metadata_yaml_to_html(meta_yaml, dataset_name=ds_name), unsafe_allow_html=True)
    else:
        st.info(
            f"No `metadata.yaml` for **{ds_name}**. "
            f"Re-run `python download_pmlb_datasets.py` to fetch dataset docs."
        )

    if df_enc is not None:
        n_feat = int(df_enc.shape[1] - 1)
        n_cls = int(df_enc["target"].nunique())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows (encoded)", f"{df_enc.shape[0]:,}")
        m2.metric("Features", f"{n_feat:,}")
        m3.metric("Classes", n_cls)
        manifest_row = catalog[catalog["dataset"].astype(str) == ds_name]
        if not manifest_row.empty and "imbalance" in manifest_row.columns:
            imb = manifest_row.iloc[0].get("imbalance")
            m4.metric("Class imbalance", f"{float(imb):.4f}" if pd.notna(imb) else "—")
        else:
            m4.metric("Columns (incl. target)", df_enc.shape[1])

    readme_body = clean_readme_markdown(readme) if readme else ""
    if readme_body:
        with st.expander("README notes", expanded=False):
            st.markdown(readme_body)

    if df_enc is not None:
        with st.expander("Data preview (first 20 rows)", expanded=False):
            st.dataframe(df_enc.head(20), use_container_width=True, hide_index=True)

    st.divider()
    _render_pycol_metrics_panel(
        dataset_name=ds_name,
        df_enc=df_enc,
        stored_row=stored_row,
        complexity_summary=complexity_summary,
        key_prefix="browse",
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
        compare_missing = st.selectbox(
            "Missing values",
            options=list(MISSING_VALUE_STRATEGIES),
            index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
            format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
            key="pmlb_cmp_missing",
        )
        use_precomputed = st.checkbox(
            "Use pre-computed PyCol metrics from `datasets_complexity_summary.csv` when available",
            value=True,
            key="pmlb_cmp_use_stored",
        )

        rows_info = []
        for name in picked:
            try:
                df_i = load_encoded_dataset(paths["root"], name)
                x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=compare_missing)
                info: dict[str, Any] = {
                    "dataset": name,
                    "pycol_precomputed": dataset_name_to_file_key(name) in pycol_done,
                    **summary_row_for_dataset(summary, name),
                    **basic_info_row(df_i, x_i, y_i, "target", missing_values=compare_missing),
                }
                rows_info.append(info)
            except Exception as exc:
                rows_info.append({"dataset": name, "error": str(exc)})
        st.subheader("Side-by-side summary")
        st.dataframe(pd.DataFrame(rows_info), use_container_width=True, hide_index=True)

        cmp_libs = st.multiselect(
            "Libraries (comparison)",
            options=["pycol", "pymfe"],
            default=["pycol"],
            key="pmlb_cmp_libs",
        )
        cmp_metric_config = render_metric_selection_block(
            cmp_libs,
            key_prefix="pmlb_cmp",
            use_all_label="Use all metrics",
        )
        cmp_recompute = st.checkbox(
            "Recompute metrics that are already stored",
            value=False,
            key="pmlb_cmp_recompute",
        )

        if st.button("Build comparison table", key="pmlb_compare_build"):
            metric_rows: list[dict[str, Any]] = []
            requested_pycol = cmp_metric_config.selected_by_library.get("pycol", [])
            progress = st.progress(0.0, text="Building comparison…")
            for i, name in enumerate(picked, start=1):
                progress.progress(i / len(picked), text=f"{name}…")
                row: dict[str, Any] = {"dataset": name, "dataset_name": name}
                stored = lookup_complexity_row(complexity_summary, name) if use_precomputed else None
                if stored is not None:
                    row.update(stored.to_dict())

                need_compute = False
                if "pycol" in cmp_libs:
                    if cmp_recompute or stored is None:
                        need_compute = bool(requested_pycol)
                    else:
                        miss = missing_pycol_keys(stored, requested_pycol)
                        need_compute = bool(miss)

                if need_compute or "pymfe" in cmp_libs:
                    try:
                        df_i = load_encoded_dataset(paths["root"], name)
                        x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=compare_missing)
                        sel = cmp_metric_config.selected_by_library
                        if "pycol" in cmp_libs and need_compute:
                            to_run = list(requested_pycol)
                            if stored is not None and not cmp_recompute:
                                to_run = [
                                    k.replace("pycol_", "")
                                    for k in missing_pycol_keys(stored, requested_pycol)
                                ]
                            if to_run:
                                row.update(
                                    compute_pycol_metrics(
                                        x_i,
                                        y_i,
                                        to_run,
                                        preset=cmp_metric_config.pycol_preset,
                                    )
                                )
                        if "pymfe" in cmp_libs:
                            row.update(
                                compute_pymfe_metrics(x_i, y_i, sel.get("pymfe", []))
                            )
                    except Exception as exc:
                        row["error"] = str(exc)
                metric_rows.append(row)
            progress.empty()
            if metric_rows:
                wide = pd.DataFrame(metric_rows)
                st.session_state["pmlb_compare_metrics"] = wide
                st.success("Comparison table ready.")

        if "pmlb_compare_metrics" in st.session_state:
            wide = st.session_state["pmlb_compare_metrics"]
            metric_cols = infer_comparison_metric_columns(wide)
            if metric_cols:
                plot_metrics = st.multiselect(
                    "Metrics to plot",
                    options=metric_cols,
                    default=metric_cols[: min(8, len(metric_cols))],
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
            with st.expander("Comparison table", expanded=False):
                st.dataframe(wide, use_container_width=True, hide_index=True)
            render_save_results_section(
                wide,
                base_name="pmlb_comparison",
                label="comparison results",
            )

        show_tsne_grid = st.checkbox("Show t-SNE grid", key="pmlb_tsne_grid")
        if show_tsne_grid and picked:
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
                    x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=compare_missing)
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
                    ax.text(0.5, 0.5, str(exc)[:60], ha="center", va="center", transform=ax.transAxes)
            for ax in axes_flat[len(picked) :]:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
