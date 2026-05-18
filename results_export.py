"""Save and download complexity result tables from Streamlit."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_results_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def render_save_results_section(
    df: pd.DataFrame,
    *,
    default_filename: str,
    key_prefix: str,
    results_dir: Path | None = None,
    show_preview: bool = True,
) -> None:
    """
    Download button + optional write to ``results/`` (persists while session state is kept).
    """
    if df is None or df.empty:
        st.caption("No results to save yet.")
        return

    out_dir = results_dir or DEFAULT_RESULTS_DIR
    st.subheader("Save results")
    st.caption(
        f"**{len(df)}** row(s), **{len(df.columns)}** columns. "
        "Download to your machine or save a copy under the project `results/` folder."
    )

    if show_preview:
        st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    safe_name = default_filename if default_filename.endswith(".csv") else f"{default_filename}.csv"

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=safe_name,
        mime="text/csv",
        key=f"{key_prefix}_download_csv",
        type="primary",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        disk_name = st.text_input(
            "Filename for project copy",
            value=safe_name,
            key=f"{key_prefix}_disk_filename",
        )
    with col_b:
        st.write("")
        st.write("")
        if st.button("Save copy to results/", key=f"{key_prefix}_save_disk"):
            fname = disk_name.strip() or safe_name
            if not fname.endswith(".csv"):
                fname = f"{fname}.csv"
            path = save_results_csv(df, out_dir / fname)
            st.success(f"Saved: `{path}`")

    with st.expander("Save with timestamped name", expanded=False):
        ts_name = f"{Path(safe_name).stem}_{_timestamp_slug()}.csv"
        if st.button(f"Save as `{ts_name}`", key=f"{key_prefix}_save_ts"):
            path = save_results_csv(df, out_dir / ts_name)
            st.success(f"Saved: `{path}`")
