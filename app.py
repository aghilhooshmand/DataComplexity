from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Dataset Complexity App", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.hero {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    margin-bottom: 1rem;
}
.card {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    margin-bottom: 0.7rem;
}
.muted {
    color: #475569;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1 style="margin:0;">📊 Dataset Complexity App</h1>
  <p style="margin:0.45rem 0 0 0;">
    A modern Streamlit workspace to compute and compare dataset complexity using <b>PyCol</b> and <b>PyMFE</b>.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
### Workspace Pages

- **`Complexity Calculator`**  
  Single-dataset complexity analysis with Upload/UCI/OpenML input, **missing-value handling** (drop rows / zero / median or mean impute), flexible metrics, CSV output, and t-SNE visualization.

- **`Dataset Comparison`**  
  Add multiple datasets, same **missing-value** options for all listed sets, compute selected complexity metrics, export comparison CSV, grouped bar charts, and optional t-SNE grid.

- **`Compare Uploaded Results`**  
  Merge pre-computed complexity CSVs (CLI or exports), download a merged table, and compare metrics with grouped bar charts.

- **`Metric Reference`**  
  PyCol overlap taxonomy (from [pycol](https://github.com/miriamspsantos/pycol/blob/doc/README.md)), **illustration PDFs**, **sample datasets** (add to comparison), and app metric tables.

- **`PMLB Benchmark`**  
  Browse [PMLB](https://github.com/EpistasisLab/pmlb) datasets in `pmlb_DS/`, view metadata and **pre-computed PyCol metrics** (`datasets_complexity_summary.csv`), compare datasets, and optional t-SNE.

**Command-line:** use `parallel_complexity_cli.py` with `--missing-values` for the same strategies on a server (see `README.md`).  
**PMLB batch download:** `python download_pmlb_datasets.py` (classification, &lt; 21k instances).
"""
)

st.markdown(
    """
<div class="card">
  <h4 style="margin:0 0 0.4rem 0;">👥 About</h4>
  <p style="margin:0.2rem 0;"><b>Maintainer:</b> Aghil Hooshmand</p>
  <p style="margin:0.2rem 0;"><b>Role:</b> Research Fellow, FORGE project (Federated Offline Reflection Grammatical Evolution)</p>
  <p style="margin:0.2rem 0;"><b>Group:</b> Biocomputing and Development Systems (BDS) Group, University of Limerick</p>
  <p style="margin:0.2rem 0;"><b>Department:</b> Computer Science and Information Systems (CSIS), UL</p>
  <p style="margin:0.2rem 0;"><b>Institute:</b> Lero, the Irish Software Engineering Research Institute</p>
  <p style="margin:0.2rem 0;"><b>Contact:</b> aghil.hooshmand@ul.ie</p>
  <p class="muted" style="margin:0.45rem 0 0 0;">
    The Biocomputing and Development Systems Group at the University of Limerick is a multi-disciplinary group that
    works with cutting-edge Artificial Intelligence and Machine Learning tools. The BDS is led by Prof. Conor Ryan,
    who invented the popular Grammatical Evolution system in 1998 with two of his PhD students.
  </p>
  <p class="muted" style="margin:0.45rem 0 0 0;">
    For collaboration, reproducibility support, or metric interpretation discussions, please get in touch at the email above.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

