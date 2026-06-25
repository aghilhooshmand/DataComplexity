from __future__ import annotations

import re
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from complexity_core import (
    MISSING_VALUE_LABELS,
    MISSING_VALUE_STRATEGIES,
    basic_info_row,
    compute_pymfe_metrics,
    compute_pycol_metrics,
    prepare_xy,
    run_tsne,
    subsample_xy_for_complexity,
)
from metric_ui import (
    MetricSelectionConfig,
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    metric_display_name,
    prepare_wide_df_for_metric_charts,
    render_metric_selection_block,
    render_per_metric_bar_charts,
)
from results_export import render_save_results_section


def extract_last_int(text: str) -> int | None:
    matches = re.findall(r"\d+", str(text))
    if not matches:
        return None
    return int(matches[-1])


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._") or "dataset"


def comparison_load_identity(source: str, loaded_name: str) -> str:
    """Stable id for the dataset currently on screen (used to reset display-name widget)."""
    if source == "Upload CSV":
        uf = st.session_state.get("cmp_upload")
        if uf is None:
            return "upload|none"
        return f"upload|{uf.name}|{getattr(uf, 'size', 0)}"
    if source == "UCI":
        ref = str(st.session_state.get("cmp_uci_ref", "")).strip()
        return f"uci|{ref}|{loaded_name}"
    ref = str(st.session_state.get("cmp_openml_ref", "")).strip()
    return f"openml|{ref}|{loaded_name}"


def sync_comparison_dataset_widgets(
    df: pd.DataFrame,
    loaded_name: str,
    load_identity: str,
) -> None:
    """Reset display name (and label column) when user loads a different dataset."""
    prev = st.session_state.get("cmp_load_identity")
    if prev == load_identity:
        return
    st.session_state["cmp_load_identity"] = load_identity
    st.session_state["cmp_custom_name"] = loaded_name
    cols = list(df.columns)
    if "target" in cols:
        st.session_state["cmp_label_col"] = "target"
    elif cols:
        st.session_state["cmp_label_col"] = cols[-1]


def load_upload_dataset() -> tuple[pd.DataFrame | None, str]:
    file = st.file_uploader("Upload CSV dataset", type=["csv"], key="cmp_upload")
    if file is None:
        return None, "dataset"
    return pd.read_csv(file), sanitize_filename(file.name.rsplit(".", 1)[0])


def load_uci_dataset() -> tuple[pd.DataFrame | None, str]:
    from ucimlrepo import fetch_ucirepo

    mode = st.radio("UCI input mode", ["Use ID", "Use Link"], horizontal=True, key="cmp_uci_mode")
    ref = st.text_input(
        "UCI dataset id" if mode == "Use ID" else "UCI dataset link",
        placeholder="53" if mode == "Use ID" else "https://archive.ics.uci.edu/dataset/53/iris",
        key="cmp_uci_ref",
    )
    if not ref:
        return None, "uci_dataset"
    ds_id = extract_last_int(ref)
    if ds_id is None:
        st.error("Could not detect UCI dataset id.")
        return None, "uci_dataset"
    try:
        ds = fetch_ucirepo(id=int(ds_id))
        x = ds.data.features
        y = ds.data.targets
        if y is None:
            st.error("UCI dataset has no target column.")
            return None, f"uci_{ds_id}"
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        df = x.copy()
        df["target"] = y
        return df, f"uci_{ds_id}"
    except Exception as exc:
        st.error(f"Failed to load UCI dataset: {exc}")
        return None, f"uci_{ds_id}"


def load_openml_dataset() -> tuple[pd.DataFrame | None, str]:
    import openml

    mode = st.radio("OpenML input mode", ["Use ID", "Use Link"], horizontal=True, key="cmp_openml_mode")
    ref = st.text_input(
        "OpenML dataset id" if mode == "Use ID" else "OpenML dataset link",
        placeholder="61" if mode == "Use ID" else "https://www.openml.org/d/61",
        key="cmp_openml_ref",
    )
    if not ref:
        return None, "openml_dataset"
    ds_id = extract_last_int(ref)
    if ds_id is None:
        st.error("Could not detect OpenML dataset id.")
        return None, "openml_dataset"
    try:
        ds = openml.datasets.get_dataset(int(ds_id))
        x, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        if y is None:
            st.error("OpenML dataset has no default target.")
            return None, f"openml_{ds_id}"
        df = x.copy()
        df["target"] = y
        return df, f"openml_{ds_id}"
    except Exception as exc:
        st.error(f"Failed to load OpenML dataset: {exc}")
        return None, f"openml_{ds_id}"


def compute_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    label_col: str,
    selected_libraries: list[str],
    selected_by_library: dict[str, list[str]],
    *,
    missing_values: str,
    complexity_max_rows: int = 0,
    pycol_matrix_mode: str = "skip",
    pycol_skip_distance_matrix: bool = True,
    pycol_parallel_heom: bool = False,
    pycol_preset: str | None = None,
    pycol_progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    x, y, _ = prepare_xy(df, label_col=label_col, missing_values=missing_values)
    rec = {"dataset_name": dataset_name}
    rec.update(basic_info_row(df, x, y, label_col, missing_values=missing_values))
    xc, yc, cmeta = subsample_xy_for_complexity(x, y, int(complexity_max_rows))
    if int(complexity_max_rows) > 0:
        rec["complexity_max_rows"] = int(complexity_max_rows)
    if cmeta.get("complexity_subsampled"):
        rec["complexity_subsampled"] = True
        rec["n_rows_complexity_input"] = int(cmeta["n_rows_complexity_input"])
        rec["n_rows_complexity_used"] = int(cmeta["n_rows_complexity_used"])
    if "pycol" in selected_libraries:
        if pycol_preset:
            rec["pycol_metrics_preset"] = pycol_preset
        rec.update(
            compute_pycol_metrics(
                xc,
                yc,
                selected_by_library.get("pycol", []),
                matrix_mode=pycol_matrix_mode,  # type: ignore[arg-type]
                preset=pycol_preset,
                parallel_heom=pycol_parallel_heom,
                heom_n_jobs=4,
                progress_callback=pycol_progress_callback,
            )
        )
    if "pymfe" in selected_libraries:
        rec.update(compute_pymfe_metrics(xc, yc, selected_by_library.get("pymfe", [])))
    return rec


st.title("Dataset Comparison")
st.markdown(
    "Add multiple datasets, compute complexity metrics, **save or download** the comparison CSV, "
    "and view per-metric bar charts."
)

if "comparison_datasets" not in st.session_state:
    st.session_state["comparison_datasets"] = []

source = st.radio("Dataset source", ["Upload CSV", "UCI", "OpenML"], horizontal=True, key="cmp_source")

df: pd.DataFrame | None = None
loaded_name = "dataset"
if source == "Upload CSV":
    df, loaded_name = load_upload_dataset()
elif source == "UCI":
    df, loaded_name = load_uci_dataset()
else:
    df, loaded_name = load_openml_dataset()

if df is not None:
    load_id = comparison_load_identity(source, loaded_name)
    sync_comparison_dataset_widgets(df, loaded_name, load_id)
    st.markdown(f"- **Loaded dataset:** `{loaded_name}`")
    st.markdown(f"- **Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
    label_col = st.selectbox(
        "Label/target column for this dataset",
        options=list(df.columns),
        key="cmp_label_col",
    )
    custom_name = st.text_input(
        "Dataset display name",
        key="cmp_custom_name",
        help="Filled automatically when you load or switch dataset (UCI id, upload file, OpenML id). "
        "Edit before **Add dataset** if you want a custom label.",
    )
    if st.button("Add dataset to comparison list", key="cmp_add_ds"):
        ds_key = sanitize_filename(custom_name)
        existing = {d["dataset_name"] for d in st.session_state["comparison_datasets"]}
        if ds_key in existing:
            st.error(
                f"Dataset name `{ds_key}` is already in the list. "
                "Change **Dataset display name** so each row has a unique name (e.g. `uci_17`, `uci_186`)."
            )
        else:
            st.session_state["comparison_datasets"].append(
                {
                    "dataset_name": ds_key,
                    "df": df.copy(),
                    "label_col": label_col,
                    "source": source,
                }
            )
            st.success(f"Added dataset: `{ds_key}`")

datasets = st.session_state["comparison_datasets"]
st.subheader("Datasets for comparison")
if datasets:
    listing = pd.DataFrame(
        [
            {
                "dataset_name": d["dataset_name"],
                "source": d["source"],
                "label_col": d["label_col"],
                "rows": int(d["df"].shape[0]),
                "columns": int(d["df"].shape[1]),
            }
            for d in datasets
        ]
    )
    st.dataframe(listing, use_container_width=True, hide_index=True)
    if st.button("Clear all comparison datasets", key="cmp_clear"):
        st.session_state["comparison_datasets"] = []
        st.rerun()
else:
    st.caption("No datasets added yet.")

missing_values = st.selectbox(
    "Missing values in features (after encoding)",
    options=list(MISSING_VALUE_STRATEGIES),
    index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
    format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
    help="Used for all datasets in the list when computing metrics or t-SNE. Rows with missing labels are always dropped.",
    key="cmp_missing_values",
)

cmp_complexity_max_rows = st.number_input(
    "Max rows for complexity (per dataset, PyCol / PyMFE)",
    min_value=0,
    max_value=2_000_000,
    value=0,
    step=100,
    help="Same as Calculator: **0** = all rows; e.g. **3000** speeds large sets (approximate metrics on a random subset).",
    key="cmp_complexity_max_rows",
)

selected_libraries = st.multiselect(
    "Complexity libraries",
    options=["pycol", "pymfe"],
    default=["pycol", "pymfe"],
    key="cmp_libs",
)

st.subheader("Metrics to compute")
_cmp_n_rows_warn = 0
if datasets:
    _cmp_n_rows_warn = max(int(d["df"].shape[0]) for d in datasets)
if int(cmp_complexity_max_rows) > 0 and _cmp_n_rows_warn > 0:
    _cmp_n_rows_warn = min(_cmp_n_rows_warn, int(cmp_complexity_max_rows))

metric_config = render_metric_selection_block(
    selected_libraries,
    key_prefix="cmp",
    use_all_label="Use all metrics",
    n_rows_for_warnings=_cmp_n_rows_warn,
)
selected_by_library = metric_config.selected_by_library

if st.button("Compute comparison metrics", type="primary", key="cmp_compute"):
    if not datasets:
        st.error("Add at least one dataset first.")
    elif not selected_libraries:
        st.error("Select at least one complexity library.")
    else:
        for lib, mlist in selected_by_library.items():
            if not mlist:
                st.error(f"Select at least one {lib} metric (or enable **Use all metrics**).")
                st.stop()
        progress = st.progress(0, text="Starting comparison computation...")
        rows: list[dict[str, Any]] = []
        total = len(datasets)
        for i, ds in enumerate(datasets, start=1):
            try:

                def _pycol_cb(
                    metric: str,
                    metric_idx: int = 0,
                    n_metrics: int = 0,
                    *,
                    _i: int = i,
                    _name: str = ds["dataset_name"],
                    _ds_total: int = total,
                ) -> None:
                    ds_pct = min(99, int((_i - 1) / _ds_total * 100 + 2))
                    if metric == "__init__":
                        progress.progress(
                            ds_pct,
                            text=f"[{_i}/{_ds_total}] `{_name}`: PyCol fast metrics…",
                        )
                    elif metric == "__init_dist__":
                        progress.progress(
                            ds_pct,
                            text=f"[{_i}/{_ds_total}] `{_name}`: PyCol building HEOM…",
                        )
                    elif metric.startswith("done:"):
                        mname = metric[5:]
                        left = max(0, n_metrics - metric_idx) if n_metrics else 0
                        progress.progress(
                            min(99, int((_i - 1) / _ds_total * 100 + 5)),
                            text=(
                                f"[{_i}/{_ds_total}] `{_name}`: "
                                f"{metric_display_name(mname, 'pycol')} done "
                                f"[{metric_idx}/{n_metrics}, {left} left]"
                            ),
                        )
                    else:
                        left = max(0, n_metrics - metric_idx + 1) if n_metrics else 0
                        progress.progress(
                            min(99, int((_i - 1) / _ds_total * 100 + 5)),
                            text=(
                                f"[{_i}/{_ds_total}] `{_name}`: "
                                f"{metric_display_name(metric, 'pycol')} "
                                f"[{metric_idx}/{n_metrics}, {left} to go]"
                            ),
                        )

                pcb = _pycol_cb if "pycol" in selected_libraries else None
                rows.append(
                    compute_for_dataset(
                        dataset_name=ds["dataset_name"],
                        df=ds["df"],
                        label_col=ds["label_col"],
                        selected_libraries=selected_libraries,
                        selected_by_library=selected_by_library,
                        missing_values=missing_values,
                        pycol_matrix_mode=metric_config.pycol_matrix_mode,
                        pycol_parallel_heom=metric_config.pycol_parallel_heom,
                        pycol_preset=metric_config.pycol_preset,
                        complexity_max_rows=int(cmp_complexity_max_rows),
                        pycol_progress_callback=pcb,
                    )
                )
            except Exception as exc:
                rows.append({"dataset_name": ds["dataset_name"], "error": str(exc)})
            pct = int((i / total) * 100)
            progress.progress(pct, text=f"Processed {i}/{total} datasets")

        out_df = pd.DataFrame(rows)
        st.session_state["comparison_result_df"] = out_df
        st.session_state["comparison_result_filename"] = "datasets_complexity_comparison.csv"
        st.success(f"Computed metrics for **{len(out_df)}** dataset(s). Save or download below.")

if "comparison_result_df" in st.session_state:
    res_df = st.session_state["comparison_result_df"]
    render_save_results_section(
        res_df,
        default_filename=str(
            st.session_state.get("comparison_result_filename", "datasets_complexity_comparison.csv")
        ),
        key_prefix="cmp_save",
    )
    metric_cols = infer_comparison_metric_columns(res_df)
    if metric_cols and "dataset_name" in res_df.columns:
        st.subheader("Metric comparison bar chart")
        chart_rows = res_df
        if "error" in res_df.columns:
            err = res_df["error"]
            failed_mask = err.notna() & err.astype(str).str.strip().ne("")
            failed = res_df[failed_mask]
            if not failed.empty:
                st.warning(
                    "Skipping datasets with computation errors in charts: "
                    + ", ".join(f"`{n}`" for n in failed["dataset_name"].astype(str))
                )
            chart_rows = res_df[~failed_mask]
        n_result_rows = len(chart_rows)
        n_unique_names = int(chart_rows["dataset_name"].nunique()) if n_result_rows else 0
        st.caption(
            f"Results table: **{n_result_rows}** row(s), **{n_unique_names}** unique `dataset_name` "
            f"(charts need one distinct name per row)."
        )
        if n_result_rows > 0 and n_unique_names < n_result_rows:
            st.error(
                f"Duplicate dataset names: **{n_result_rows}** rows but only **{n_unique_names}** "
                "unique name(s). Bars were drawn on top of each other. "
                "Clear the list, re-add each dataset with a **unique display name**, then recompute."
            )
            with st.expander("Rows in results (dataset_name)"):
                st.dataframe(
                    chart_rows[["dataset_name"]].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )
        plot_metrics = st.multiselect(
            "Metrics to plot",
            options=metric_cols,
            default=metric_cols[: min(8, len(metric_cols))],
            key="cmp_plot_metrics",
        )
        if plot_metrics:
            _, chart_warnings = prepare_wide_df_for_metric_charts(
                chart_rows, dataset_field="dataset_name", metric_columns=plot_metrics
            )
            for msg in chart_warnings:
                st.warning(msg)
            long_df = melt_metrics_for_comparison(
                chart_rows, dataset_field="dataset_name", metric_columns=plot_metrics
            )
            render_per_metric_bar_charts(
                long_df,
                dataset_field="dataset_name",
                metrics_order=plot_metrics,
            )

st.subheader("t-SNE comparison")
show_tsne_compare = st.checkbox(
    "Also compare t-SNE for datasets in comparison list",
    value=False,
    key="cmp_show_tsne_compare",
)
if show_tsne_compare:
    if not datasets:
        st.caption("Add datasets to the comparison list first.")
    elif st.button("Generate t-SNE grid", key="cmp_generate_tsne_grid"):
        cols = st.columns(2)
        for i, ds in enumerate(datasets):
            with cols[i % 2]:
                st.markdown(f"**{ds['dataset_name']}**")
                try:
                    x_tsne, y_tsne, _ = prepare_xy(
                        ds["df"],
                        label_col=ds["label_col"],
                        missing_values=missing_values,
                    )
                    tsne_df = run_tsne(x_tsne, y_tsne)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    scatter = ax.scatter(
                        tsne_df["tsne_1"],
                        tsne_df["tsne_2"],
                        c=tsne_df["label_code"],
                        cmap="tab10",
                        s=14,
                        alpha=0.85,
                    )
                    ax.set_title(f"t-SNE: {ds['dataset_name']}")
                    ax.set_xlabel("t-SNE 1")
                    ax.set_ylabel("t-SNE 2")
                    fig.colorbar(scatter, ax=ax, label="Label code")
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Failed for {ds['dataset_name']}: {exc}")
