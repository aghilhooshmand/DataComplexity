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
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    prepare_wide_df_for_metric_charts,
    render_metric_selection_block,
    render_per_metric_bar_charts,
)
from pmlb_io import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUMMARY_TSV,
    clean_readme_markdown,
    load_cached_metadata_yaml,
    load_cached_readme,
    load_encoded_dataset,
    load_manifest,
    load_summary_table,
    metadata_dir_for_dataset,
    metadata_yaml_to_html,
    output_paths,
    select_benchmark_datasets,
    summary_row_for_dataset,
)
from results_export import render_save_results_section


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

manifest = load_manifest(paths["root"])
summary = load_summary_table(DEFAULT_SUMMARY_TSV)
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
    st.caption(
        f"{len(pmlb_available)} classification datasets with n_instances < 21,000 in the summary file; "
        f"{len(dataset_names)} encoded locally."
    )
    if paths["status"].is_file():
        st.markdown("**Last download run**")
        st.json(paths["status"].read_text(encoding="utf-8"))

with tab_browse:
    ds_name = st.selectbox("Dataset", dataset_names, key="pmlb_ds_select")
    meta_dir = metadata_dir_for_dataset(paths["root"], ds_name)
    meta_yaml = load_cached_metadata_yaml(meta_dir)
    readme = load_cached_readme(meta_dir)

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
    st.subheader("Complexity & visualization")
    missing_values = st.selectbox(
        "Missing values (re-encode from CSV for analysis)",
        options=list(MISSING_VALUE_STRATEGIES),
        index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
        format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
        key="pmlb_missing_values",
    )
    complexity_max_rows = st.number_input(
        "Max rows for PyCol / PyMFE (0 = all)",
        min_value=0,
        max_value=500_000,
        value=0,
        step=500,
        key="pmlb_max_rows",
    )
    selected_libraries = st.multiselect(
        "Libraries",
        options=["pycol", "pymfe"],
        default=["pycol"],
        key="pmlb_libs",
    )
    metric_config = render_metric_selection_block(
        selected_libraries,
        key_prefix="pmlb",
        use_all_label="Use all metrics",
        n_rows_for_warnings=int(df_enc.shape[0]) if df_enc is not None else 0,
    )

    if df_enc is not None and st.button("Compute complexity for this dataset", key="pmlb_compute_one"):
        try:
            x, y, _ = prepare_xy(df_enc, "target", missing_values=missing_values)
            x, y, sub_meta = subsample_xy_for_complexity(
                x, y, int(complexity_max_rows), random_state=0
            )
            row: dict[str, Any] = {
                "dataset": ds_name,
                **basic_info_row(df_enc, x, y, "target", missing_values=missing_values),
                **sub_meta,
            }
            selected_by_library = metric_config.selected_by_library
            if "pycol" in selected_libraries:
                row.update(
                    compute_pycol_metrics(
                        x,
                        y,
                        selected_by_library.get("pycol", []),
                        preset=metric_config.pycol_preset,
                    )
                )
            if "pymfe" in selected_libraries:
                row.update(
                    compute_pymfe_metrics(x, y, selected_by_library.get("pymfe", []))
                )
            st.session_state["pmlb_last_metrics"] = row
            st.success("Complexity computed.")
        except Exception as exc:
            st.error(f"Computation failed: {exc}")

    if "pmlb_last_metrics" in st.session_state:
        st.json(st.session_state["pmlb_last_metrics"])

    if paths["complexity_cache"].is_file():
        st.markdown("**Pre-computed batch metrics** (`complexity_metrics.csv`)")
        batch_df = pd.read_csv(paths["complexity_cache"])
        one = batch_df[batch_df["dataset"].astype(str) == ds_name]
        if not one.empty:
            st.dataframe(one.T, use_container_width=True)
        else:
            st.caption("This dataset is not in the batch complexity file yet.")

    if df_enc is not None and st.button("Show t-SNE", key="pmlb_tsne_one"):
        try:
            x, y, _ = prepare_xy(df_enc, "target", missing_values=missing_values)
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
        rows_info = []
        for name in picked:
            try:
                df_i = load_encoded_dataset(paths["root"], name)
                x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=compare_missing)
                rows_info.append(
                    {
                        "dataset": name,
                        **summary_row_for_dataset(summary, name),
                        **basic_info_row(df_i, x_i, y_i, "target", missing_values=compare_missing),
                    }
                )
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

        if st.button("Compute & compare complexity", key="pmlb_compare_compute"):
            metric_rows: list[dict[str, Any]] = []
            progress = st.progress(0.0, text="Computing…")
            for i, name in enumerate(picked, start=1):
                progress.progress(i / len(picked), text=f"{name}…")
                try:
                    df_i = load_encoded_dataset(paths["root"], name)
                    x_i, y_i, _ = prepare_xy(df_i, "target", missing_values=compare_missing)
                    row: dict[str, Any] = {"dataset": name}
                    sel = cmp_metric_config.selected_by_library
                    if "pycol" in cmp_libs:
                        row.update(
                            compute_pycol_metrics(
                                x_i,
                                y_i,
                                sel.get("pycol", []),
                                preset=cmp_metric_config.pycol_preset,
                            )
                        )
                    if "pymfe" in cmp_libs:
                        row.update(
                            compute_pymfe_metrics(x_i, y_i, sel.get("pymfe", []))
                        )
                    metric_rows.append(row)
                except Exception as exc:
                    st.warning(f"{name}: {exc}")
            progress.empty()
            if metric_rows:
                wide = pd.DataFrame(metric_rows)
                st.session_state["pmlb_compare_metrics"] = wide
                st.success("Comparison metrics ready.")

        if "pmlb_compare_metrics" in st.session_state:
            wide = st.session_state["pmlb_compare_metrics"]
            metric_cols = infer_comparison_metric_columns(wide)
            if metric_cols:
                chart_df = prepare_wide_df_for_metric_charts(wide, metric_cols)
                melted = melt_metrics_for_comparison(chart_df, metric_cols)
                render_per_metric_bar_charts(melted, metric_cols)
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
