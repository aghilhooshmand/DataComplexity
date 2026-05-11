from __future__ import annotations

import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from complexity_core import (
    MISSING_VALUE_LABELS,
    MISSING_VALUE_STRATEGIES,
    available_metrics_by_library,
    basic_info_row,
    compute_pymfe_metrics,
    compute_pycol_metrics,
    prepare_xy,
    run_tsne,
)
from metric_catalog import PYMFE_COMPLEXITY_METRICS, PYCOL_METRICS


def extract_last_int(text: str) -> int | None:
    matches = re.findall(r"\d+", str(text))
    if not matches:
        return None
    return int(matches[-1])


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._") or "dataset"


def load_upload_dataset() -> tuple[pd.DataFrame | None, str, dict]:
    file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if file is None:
        return None, "dataset", {}
    dataset_name = sanitize_filename(file.name.rsplit(".", 1)[0])
    meta = {
        "source": "Upload CSV",
        "dataset_name": file.name,
    }
    return pd.read_csv(file), dataset_name, meta


def load_uci_dataset() -> tuple[pd.DataFrame | None, str, dict]:
    from ucimlrepo import fetch_ucirepo

    uci_mode = st.radio("UCI input mode", options=["Use ID", "Use Link"], horizontal=True)
    if uci_mode == "Use ID":
        uci_ref = st.text_input("UCI dataset id", placeholder="53")
    else:
        uci_ref = st.text_input(
            "UCI dataset link",
            placeholder="https://archive.ics.uci.edu/dataset/53/iris",
        )
    if not uci_ref:
        return None, "uci_dataset", {}
    uci_id = extract_last_int(uci_ref)
    if uci_id is None:
        st.error("Could not detect a numeric UCI dataset id from your input.")
        return None, "uci_dataset", {}
    try:
        ds = fetch_ucirepo(id=int(uci_id))
        x = ds.data.features
        y = ds.data.targets
        if y is None:
            st.error("UCI dataset has no target column.")
            return None, f"uci_{uci_id}", {}
        if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
            y_col = y.columns[0]
            y_series = y[y_col]
        elif isinstance(y, pd.DataFrame):
            y_series = y.iloc[:, 0]
        else:
            y_series = y
        df = x.copy()
        df["target"] = y_series
        md = getattr(ds, "metadata", {}) or {}
        meta = {
            "source": "UCI",
            "dataset_name": md.get("name", f"uci_{uci_id}"),
            "dataset_id": uci_id,
            "published_date": md.get("donated_date", md.get("publication_year")),
            "task": md.get("task"),
            "data_type": md.get("data_type"),
            "n_instances_meta": md.get("num_instances"),
            "n_features_meta": md.get("num_features"),
            "url": md.get("repository_url"),
        }
        return df, f"uci_{uci_id}", meta
    except Exception as exc:
        st.error(f"Failed to load UCI dataset: {exc}")
        return None, f"uci_{uci_id}", {}


def load_openml_dataset() -> tuple[pd.DataFrame | None, str, dict]:
    import openml

    openml_mode = st.radio("OpenML input mode", options=["Use ID", "Use Link"], horizontal=True)
    if openml_mode == "Use ID":
        dataset_ref = st.text_input("OpenML dataset id", placeholder="61")
    else:
        dataset_ref = st.text_input(
            "OpenML dataset link",
            placeholder="https://www.openml.org/d/61",
        )
    if not dataset_ref:
        return None, "openml_dataset", {}
    dataset_id = extract_last_int(dataset_ref)
    if dataset_id is None:
        st.error("Could not detect a numeric OpenML dataset id from your input.")
        return None, "openml_dataset", {}
    try:
        ds = openml.datasets.get_dataset(int(dataset_id))
        x, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        if y is None:
            st.error("OpenML dataset has no default target attribute.")
            return None, f"openml_{dataset_id}", {}
        df = x.copy()
        df["target"] = y
        meta = {
            "source": "OpenML",
            "dataset_name": getattr(ds, "name", f"openml_{dataset_id}"),
            "dataset_id": dataset_id,
            "published_date": getattr(ds, "upload_date", None),
            "default_target": getattr(ds, "default_target_attribute", None),
            "version": getattr(ds, "version", None),
            "description": getattr(ds, "description", None),
            "url": f"https://www.openml.org/d/{dataset_id}",
        }
        return df, f"openml_{dataset_id}", meta
    except Exception as exc:
        st.error(f"Failed to load OpenML dataset: {exc}")
        return None, f"openml_{dataset_id}", {}


def render_metric_docs(library: str, selected: list[str]) -> None:
    docs = PYCOL_METRICS if library == "pycol" else PYMFE_COMPLEXITY_METRICS
    if not selected:
        st.caption("Select metrics to see their descriptions and references.")
        return
    for metric in selected:
        doc = docs.get(metric)
        if doc is None:
            st.markdown(f"**{metric}**")
            st.caption("No local description card. This metric is still computed and included in output.")
            continue
        st.markdown(f"**{doc.title}**")
        st.caption(doc.description)
        st.caption(f"Reference: {doc.reference}")
        st.divider()


def render_metric_docs_multilib(selected_libraries: list[str], selected_by_library: dict[str, list[str]]) -> None:
    if not selected_libraries:
        st.caption("Select at least one library to see metric info.")
        return
    for lib in selected_libraries:
        metrics = selected_by_library.get(lib, [])
        st.markdown(f"### {lib.upper()} metrics")
        if not metrics:
            st.caption("No metrics selected.")
            continue
        render_metric_docs(lib, metrics)


st.title("Complexity Calculator")
st.markdown(
    "Compute data complexity for one dataset via three sources: Upload CSV, UCI, or OpenML."
)
with st.expander("How to use this page", expanded=True):
    st.markdown(
        """
1. Choose **Dataset source**: `Upload CSV`, `UCI`, or `OpenML`.
2. For `UCI` or `OpenML`, paste either:
   - just the dataset id (example: `53`), or
   - a full dataset link (example: `https://archive.ics.uci.edu/dataset/53/iris`).
3. Select the **label/target column**.
4. Select one or both libraries (`pycol`, `pymfe`) and metric set (`all` or custom).
5. Click **Compute complexity** to get a one-row table and download CSV.
6. Click **Show t-SNE of dataset** to visualize the dataset in 2D.
"""
    )

source = st.radio(
    "Dataset source",
    options=["Upload CSV", "UCI", "OpenML"],
    horizontal=True,
)

df: pd.DataFrame | None = None
dataset_name = "dataset"
dataset_meta: dict = {}
if source == "Upload CSV":
    df, dataset_name, dataset_meta = load_upload_dataset()
elif source == "UCI":
    df, dataset_name, dataset_meta = load_uci_dataset()
else:
    df, dataset_name, dataset_meta = load_openml_dataset()

if df is not None:
    label_col = st.selectbox("Choose label/target column", options=list(df.columns))
    missing_values = st.selectbox(
        "Missing values in features (after encoding)",
        options=list(MISSING_VALUE_STRATEGIES),
        index=list(MISSING_VALUE_STRATEGIES).index("impute_median"),
        format_func=lambda k: MISSING_VALUE_LABELS.get(str(k), str(k)),
        help="Applies after turning categories into numbers. Rows with missing labels are always dropped.",
    )
    st.subheader("Dataset summary")
    summary_lines: list[str] = []
    summary_lines.append(f"- **Source:** {dataset_meta.get('source', source)}")
    summary_lines.append(f"- **Dataset name:** {dataset_meta.get('dataset_name', dataset_name)}")
    if dataset_meta.get("dataset_id") is not None:
        summary_lines.append(f"- **Dataset id:** {dataset_meta.get('dataset_id')}")
    if dataset_meta.get("published_date"):
        summary_lines.append(f"- **Published/Donated date:** {dataset_meta.get('published_date')}")
    if dataset_meta.get("task"):
        summary_lines.append(f"- **Task:** {dataset_meta.get('task')}")
    if dataset_meta.get("data_type"):
        summary_lines.append(f"- **Data type:** {dataset_meta.get('data_type')}")
    if dataset_meta.get("default_target"):
        summary_lines.append(f"- **Default target (metadata):** {dataset_meta.get('default_target')}")
    if dataset_meta.get("version") is not None:
        summary_lines.append(f"- **Version:** {dataset_meta.get('version')}")
    if dataset_meta.get("url"):
        summary_lines.append(f"- **Dataset URL:** {dataset_meta.get('url')}")
    summary_lines.append(f"- **Selected label column:** {label_col}")
    summary_lines.append(f"- **Rows (original):** {int(df.shape[0])}")
    summary_lines.append(f"- **Columns (original):** {int(df.shape[1])}")
    summary_lines.append(f"- **Missing values (original total):** {int(df.isna().sum().sum())}")
    summary_lines.append(f"- **Missing-value strategy:** `{missing_values}`")
    try:
        x_preview, y_preview, merged_preview = prepare_xy(
            df, label_col=label_col, missing_values=missing_values
        )
        summary_lines.append(f"- **Rows (after cleaning):** {int(merged_preview.shape[0])}")
        summary_lines.append(f"- **Features (after encoding):** {int(x_preview.shape[1])}")
        summary_lines.append(f"- **Classes:** {int(len(set(y_preview.tolist())))}")
    except Exception as exc:
        summary_lines.append(f"- **Preprocessing status:** Failed ({exc})")
    st.markdown("\n".join(summary_lines))

    st.success(f"Loaded dataset with shape: {df.shape}")
    st.caption("CSV preview")
    st.dataframe(df.head(20), use_container_width=True)
    selected_libraries = st.multiselect(
        "Complexity libraries",
        options=["pycol", "pymfe"],
        default=["pycol", "pymfe"],
    )

    use_all = st.checkbox("Use all metrics", value=True)

    selected_by_library: dict[str, list[str]] = {}
    for lib in selected_libraries:
        all_metrics = available_metrics_by_library(lib)
        if use_all:
            selected_by_library[lib] = all_metrics
        else:
            selected_by_library[lib] = st.multiselect(
                f"Select {lib} metrics to compute",
                options=all_metrics,
                default=all_metrics[: min(8, len(all_metrics))],
                key=f"metrics_{lib}",
            )

    with st.expander("Selected metric descriptions and references", expanded=False):
        render_metric_docs_multilib(selected_libraries, selected_by_library)

    if st.button("Compute complexity", type="primary"):
        try:
            if not selected_libraries:
                st.error("Please select at least one complexity library.")
                st.stop()

            progress = st.progress(0, text="Starting complexity computation...")
            status_box = st.empty()

            status_box.info("Step 1/4: Preparing and cleaning dataset")
            x, y, _ = prepare_xy(df, label_col=label_col, missing_values=missing_values)
            progress.progress(25, text="Prepared dataset")

            status_box.info("Step 2/4: Building dataset summary")
            result = basic_info_row(df, x, y, label_col, missing_values=missing_values)
            progress.progress(45, text="Built dataset summary")

            status_box.info(f"Step 3/4: Computing metrics with {', '.join(selected_libraries)}")
            if "pycol" in selected_libraries:
                result.update(compute_pycol_metrics(x, y, selected_by_library.get("pycol", [])))
            if "pymfe" in selected_libraries:
                result.update(compute_pymfe_metrics(x, y, selected_by_library.get("pymfe", [])))
            progress.progress(85, text="Computed selected metrics")

            status_box.info("Step 4/4: Preparing result table and download")
            out_df = pd.DataFrame([result])
            st.subheader("Complexity result (CSV row)")
            st.dataframe(out_df, use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download result CSV",
                data=csv_bytes,
                file_name=f"{sanitize_filename(dataset_name)}_complexity.csv",
                mime="text/csv",
            )
            st.session_state["latest_xy"] = (x, y)
            progress.progress(100, text="Complexity computation finished")
            status_box.success("Done: complexity metrics computed successfully.")
        except Exception as exc:
            st.error(f"Computation failed: {exc}")

    if st.button("Show t-SNE of dataset"):
        try:
            if "latest_xy" in st.session_state:
                x, y = st.session_state["latest_xy"]
            else:
                x, y, _ = prepare_xy(df, label_col=label_col, missing_values=missing_values)
            tsne_df = run_tsne(x, y)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                tsne_df["tsne_1"],
                tsne_df["tsne_2"],
                c=tsne_df["label_code"],
                cmap="tab10",
                s=18,
                alpha=0.85,
            )
            ax.set_title("t-SNE projection")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            fig.colorbar(scatter, ax=ax, label="Label code")
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                "Download t-SNE PNG",
                data=buf.getvalue(),
                file_name="dataset_tsne.png",
                mime="image/png",
            )
        except Exception as exc:
            st.error(f"t-SNE failed: {exc}")

