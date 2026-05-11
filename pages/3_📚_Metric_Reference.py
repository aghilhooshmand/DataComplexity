from __future__ import annotations

import pandas as pd
import streamlit as st

from metric_catalog import PYMFE_COMPLEXITY_METRICS, PYCOL_METRICS


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


st.title("Metric Reference")
st.markdown("Browse complexity metrics supported in this app.")

tab1, tab2 = st.tabs(["pycol", "pymfe"])
with tab1:
    st.dataframe(docs_to_df(PYCOL_METRICS), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(docs_to_df(PYMFE_COMPLEXITY_METRICS), use_container_width=True, hide_index=True)

