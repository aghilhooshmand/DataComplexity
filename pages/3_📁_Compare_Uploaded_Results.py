from __future__ import annotations

import io
import re

import pandas as pd
import streamlit as st

from metric_ui import (
    infer_comparison_metric_columns,
    melt_metrics_for_comparison,
    prepare_wide_df_for_metric_charts,
    render_per_metric_bar_charts,
)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._") or "dataset"


def merge_uploaded_frames(frames: list[tuple[str, pd.DataFrame]], id_column: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source_label, df in frames:
        piece = df.copy()
        piece["_upload_file"] = source_label
        parts.append(piece)
    merged = pd.concat(parts, ignore_index=True, sort=False)
    if id_column not in merged.columns:
        raise ValueError(f"Identifier column {id_column!r} not found after merge.")
    merged[id_column] = merged[id_column].astype(str)
    merged = merged.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)
    return merged


st.title("Compare uploaded complexity CSVs")
st.markdown(
    "Merge **pre-computed** result files (for example from `parallel_complexity_cli.py` or the **Dataset Comparison** "
    "export), preview and download a merged table, and compare metrics with grouped bar charts."
)

st.subheader("Pre-computed metrics (CSV)")
uploaded = st.file_uploader(
    "Upload one or more complexity CSV files",
    type=["csv"],
    accept_multiple_files=True,
    key="upr_results_files",
)

if uploaded:
    frames: list[tuple[str, pd.DataFrame]] = []
    errors: list[str] = []
    for uf in uploaded:
        label = sanitize_filename(uf.name.rsplit(".", 1)[0])
        try:
            raw = uf.read()
            frames.append((label, pd.read_csv(io.BytesIO(raw))))
        except Exception as exc:
            errors.append(f"{uf.name}: {exc}")

    for e in errors:
        st.error(e)

    if frames:
        all_columns: set[str] = set()
        for _, d in frames:
            all_columns.update(d.columns)
        default_id = "dataset_name" if "dataset_name" in all_columns else None
        id_options = sorted(all_columns)
        id_column = st.selectbox(
            "Column that uniquely identifies each dataset (row)",
            options=id_options,
            index=id_options.index(default_id) if default_id in id_options else 0,
            help="CLI output uses `dataset_name`. Pick another column if your files use a different key.",
            key="upr_id_col",
        )
        try:
            merged_df = merge_uploaded_frames(frames, id_column=id_column)
        except Exception as exc:
            st.error(str(exc))
            merged_df = None

        if merged_df is not None:
            st.success(f"Merged **{len(merged_df)}** dataset row(s) from **{len(frames)}** file(s).")
            st.dataframe(merged_df, use_container_width=True, hide_index=True)
            export_df = merged_df.drop(columns=["_upload_file"], errors="ignore")
            st.download_button(
                "Download merged CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="merged_complexity_uploads.csv",
                mime="text/csv",
                key="upr_download_merged",
            )

            metric_cols = infer_comparison_metric_columns(merged_df)
            if not metric_cols:
                st.warning(
                    "No numeric `pycol_*` or `pymfe_*` columns found—bar charts need those columns in the uploaded files."
                )
            elif id_column in merged_df.columns:
                st.subheader("Metric comparison bar chart")
                n_rows = len(merged_df)
                n_unique = int(merged_df[id_column].nunique())
                st.caption(
                    f"Merged table: **{n_rows}** row(s), **{n_unique}** unique `{id_column}`."
                )
                if n_unique < n_rows:
                    st.error(
                        f"Duplicate `{id_column}` values — charts will merge bars unless names are unique per row."
                    )
                plot_metrics = st.multiselect(
                    "Metrics to plot",
                    options=metric_cols,
                    default=metric_cols[: min(8, len(metric_cols))],
                    key="upr_plot_metrics",
                )
                if plot_metrics:
                    _, chart_warnings = prepare_wide_df_for_metric_charts(
                        merged_df, dataset_field=id_column, metric_columns=plot_metrics
                    )
                    for msg in chart_warnings:
                        st.warning(msg)
                    long_df = melt_metrics_for_comparison(
                        merged_df, dataset_field=id_column, metric_columns=plot_metrics
                    )
                    render_per_metric_bar_charts(
                        long_df,
                        dataset_field=id_column,
                        metrics_order=plot_metrics,
                    )
else:
    st.caption("Upload exported or CLI-written complexity CSVs to merge, download, and plot metrics.")
