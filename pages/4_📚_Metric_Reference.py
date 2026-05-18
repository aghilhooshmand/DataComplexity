from __future__ import annotations

import pandas as pd
import streamlit as st

from metric_catalog import PYMFE_COMPLEXITY_METRICS, PYCOL_METRICS
from pycol_reference import (
    PYCOL_AIRE_PAPER,
    PYCOL_DATASET_DIR,
    PYCOL_GITHUB_IMG,
    PYCOL_GITHUB_README,
    PYCOL_IMG_DIR,
    PYCOL_OVERLAP_CATEGORIES,
    list_available_figures,
    list_available_sample_datasets,
    load_pycol_sample_dataset,
    pycol_sample_dataset_name,
    render_pdf_embed,
)
from pycol_reference import figure_path as pycol_figure_path


def docs_to_df(items: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": doc.key,
                "title": doc.title,
                "description": doc.description,
                "reference": doc.reference,
            }
            for doc in items.values()
        ]
    ).sort_values("metric")


def _add_sample_to_comparison(spec_file: str) -> None:
    from pycol_reference import PYCOL_SAMPLE_DATASETS

    spec = next((s for s in PYCOL_SAMPLE_DATASETS if s.file == spec_file), None)
    if spec is None:
        st.error("Unknown sample dataset.")
        return
    try:
        df = load_pycol_sample_dataset(spec)
    except Exception as exc:
        st.error(f"Could not load `{spec.file}`: {exc}")
        return

    ds_name = pycol_sample_dataset_name(spec)
    label_col = spec.label_column if spec.label_column in df.columns else df.columns[-1]

    if "comparison_datasets" not in st.session_state:
        st.session_state["comparison_datasets"] = []

    existing = {d["dataset_name"] for d in st.session_state["comparison_datasets"]}
    if ds_name in existing:
        st.warning(f"`{ds_name}` is already in the comparison list.")
        return

    st.session_state["comparison_datasets"].append(
        {
            "dataset_name": ds_name,
            "df": df,
            "label_col": label_col,
            "source": "PyCol sample",
        }
    )
    st.success(f"Added **`{ds_name}`** ({df.shape[0]} rows) to **Dataset Comparison**. Open that page to compute metrics.")


st.title("Metric Reference")
st.markdown(
    "PyCol class-overlap measures ([library README](%s)), illustrations from the AIRE review, "
    "and **sample datasets** shipped with PyCol (`pycol-doc/dataset/`)."
    % PYCOL_GITHUB_README
)

tab_overview, tab_metrics, tab_figures, tab_datasets = st.tabs(
    ["PyCol overview", "App metrics (pycol / pymfe)", "Illustrations", "Sample datasets"]
)

with tab_overview:
    st.markdown(
        """
**pycol** (Python Class Overlap Library) groups complexity measures by four overlap views
([Santos et al., AIRE 2022](%s)):

- **Feature overlap** — separability and box/range overlap in input space (F1–F4, input noise).
- **Instance overlap** — neighbor ambiguity, densities, invasive points (N3, N4, kDN, CM, …).
- **Structural overlap** — boundaries, clusters, hypersphere coverage (N1, N2, T1, DBC, …).
- **Multiresolution overlap** — grid purity, case-base profiles (MRCA, C1, C2, purity, …).

This app runs a **subset** via `pycol-complexity` presets (`cheap_minimal`, `cheap`, …).
See the **Illustrations** tab for figures from the PyCol documentation.
        """
        % PYCOL_AIRE_PAPER
    )
    for cat in PYCOL_OVERLAP_CATEGORIES:
        with st.expander(cat.title, expanded=cat.key == "feature"):
            st.markdown(", ".join(f"**{m}**" for m in cat.measures))

with tab_metrics:
    st.caption("Metrics wired in this app (not the full PyCol catalog).")
    sub_pycol, sub_pymfe = st.tabs(["pycol", "pymfe"])
    with sub_pycol:
        st.dataframe(docs_to_df(PYCOL_METRICS), use_container_width=True, hide_index=True)
    with sub_pymfe:
        st.dataframe(docs_to_df(PYMFE_COMPLEXITY_METRICS), use_container_width=True, hide_index=True)

with tab_figures:
    st.markdown(
        "Figures from [pycol/doc/img](%s) (PDF). Use them to interpret overlap types and individual measures."
        % PYCOL_GITHUB_IMG
    )
    figures = list_available_figures()
    if not figures:
        st.warning(f"No PDF figures found under `{PYCOL_IMG_DIR}`.")
    else:
        categories = sorted({f.category for f in figures})
        cat_filter = st.selectbox("Category", options=["All"] + categories, key="pycol_fig_cat")
        shown = figures if cat_filter == "All" else [f for f in figures if f.category == cat_filter]
        labels = [f"{f.title} ({f.file})" for f in shown]
        pick = st.selectbox("Figure", options=range(len(shown)), format_func=lambda i: labels[i], key="pycol_fig_pick")
        fig = shown[int(pick)]
        st.subheader(fig.title)
        st.caption(f"**Category:** {fig.category} · **File:** `{fig.file}`")
        if fig.caption:
            st.markdown(fig.caption)
        render_pdf_embed(pycol_figure_path(fig), height=560)
        with open(pycol_figure_path(fig), "rb") as fh:
            st.download_button(
                "Download PDF",
                data=fh.read(),
                file_name=fig.file,
                mime="application/pdf",
                key=f"pycol_dl_{fig.file}",
            )

with tab_datasets:
    st.markdown(
        """
Sample datasets from the [PyCol `dataset` folder](%s): numerical features, no missing values
(binary and multi-class). Load one into **Dataset Comparison** or download the raw file.
        """
        % (PYCOL_GITHUB_README.replace("/README.md", "/tree/doc/dataset"))
    )
    samples = list_available_sample_datasets()
    if not samples:
        st.warning(f"No sample files under `{PYCOL_DATASET_DIR}`.")
    else:
        rows = []
        for s in samples:
            path = PYCOL_DATASET_DIR / s.file
            rows.append(
                {
                    "file": s.file,
                    "title": s.title,
                    "label_column": s.label_column,
                    "notes": s.notes,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Load into Dataset Comparison")
        file_options = [s.file for s in samples]
        chosen = st.selectbox(
            "Sample dataset",
            options=file_options,
            format_func=lambda f: next(s.title for s in samples if s.file == f),
            key="pycol_sample_pick",
        )
        spec = next(s for s in samples if s.file == chosen)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Add to comparison list", type="primary", key="pycol_add_cmp"):
                _add_sample_to_comparison(chosen)
        with col_b:
            try:
                preview = load_pycol_sample_dataset(spec).head(8)
                with st.expander("Preview first rows"):
                    st.dataframe(preview, use_container_width=True)
            except Exception as exc:
                st.caption(f"Preview failed: {exc}")
        with col_c:
            path = PYCOL_DATASET_DIR / chosen
            st.download_button(
                "Download file",
                data=path.read_bytes(),
                file_name=chosen,
                mime="application/octet-stream",
                key=f"pycol_dl_ds_{chosen}",
            )
