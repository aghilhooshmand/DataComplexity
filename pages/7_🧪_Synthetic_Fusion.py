from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from complexity_profiler import (
    exemplar_table,
    profile_summary,
    top_exemplars_for_metric,
)
from metric_catalog import PYCOL_METRICS, metric_direction_label
from metric_ui import metric_display_name
from pmlb_io import DEFAULT_OUTPUT_DIR, list_downloaded_datasets, pycol_metric_columns
from results_export import render_save_results_section
from summary_dashboard import resolve_complexity_summary_path
from synthetic_fusion import AnchorAugmentConfig, generate_augmented_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "results" / "synthetic"

PYCOL_BOOST_METRICS = sorted(PYCOL_METRICS.keys())


@st.cache_data(show_spinner=False)
def _cached_profile(summary_mtime_ns: int, summary_path: str, threshold: float) -> pd.DataFrame:
    return profile_summary(summary_path=summary_path, threshold=threshold)


def _summary_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


st.set_page_config(page_title="Dataset Augmentation", layout="wide")
st.title("🧪 Harder Dataset Generator")
st.markdown(
    """
Pick **one anchor dataset** (e.g. `ring`, already hard on F1). Choose **metrics to boost** (e.g. C1).
The app finds a **donor** dataset that is hardest on each metric, then generates **new samples inside
the anchor's own classes** — labels keep their anchor meaning.

CLI: `python synthetic_fusion_cli.py --anchor ring --boost-metrics C1 --output ...`
"""
)

if "scf_step" not in st.session_state:
    st.session_state["scf_step"] = 1

steps = [
    "1. Load profile",
    "2. Anchor dataset",
    "3. Metrics & donors",
    "4. Settings",
    "5. Generate & save",
]
step = st.sidebar.radio(
    "Step",
    options=list(range(1, 6)),
    format_func=lambda i: steps[i - 1],
    key="scf_step_radio",
)
st.session_state["scf_step"] = step

if step == 1:
    st.header("Step 1 — Load complexity profile")

    try:
        default_summary = resolve_complexity_summary_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    summary_path = st.text_input("Complexity summary CSV", value=str(default_summary), key="scf_summary_path")
    threshold = st.slider("Archetype threshold", 0.5, 0.95, 0.75, 0.05, key="scf_archetype_thr")

    path = Path(summary_path)
    if not path.is_file():
        st.warning("Summary file not found.")
        st.stop()

    profiled = _cached_profile(_summary_mtime(path), str(path), float(threshold))
    st.success(f"Loaded **{len(profiled)}** datasets.")

    hard_cols = [c for c in profiled.columns if c.startswith("hard_")][:8]
    show = [c for c in ["dataset_file", "archetypes", *hard_cols] if c in profiled.columns]
    st.dataframe(profiled[show].head(15), use_container_width=True, hide_index=True)

    if st.button("Continue →", type="primary", key="scf_s1_next"):
        st.session_state["scf_step"] = 2
        st.rerun()

elif step == 2:
    st.header("Step 2 — Choose anchor dataset")
    st.caption("The dataset you want to make harder. Its **class labels stay the same**.")

    pmlb_names = list_downloaded_datasets(DEFAULT_OUTPUT_DIR)
    if not pmlb_names:
        st.error("No CSV files in `pmlb_DS/`.")
        st.stop()

    summary_path = st.session_state.get("scf_summary_path", str(resolve_complexity_summary_path()))
    profiled = _cached_profile(
        _summary_mtime(Path(summary_path)),
        str(summary_path),
        float(st.session_state.get("scf_archetype_thr", 0.75)),
    )

    default_anchor = st.session_state.get("scf_anchor", pmlb_names[0])
    anchor = st.selectbox(
        "Anchor dataset",
        options=pmlb_names,
        index=pmlb_names.index(default_anchor) if default_anchor in pmlb_names else 0,
        key="scf_anchor_pick",
    )
    st.session_state["scf_anchor"] = anchor

    pycol_cols = pycol_metric_columns(profiled)
    row = profiled[profiled["dataset_file"].astype(str) == f"{anchor}.csv"]
    if not row.empty and pycol_cols:
        st.subheader(f"Current metrics for `{anchor}`")
        r = row.iloc[0]
        preview = []
        for c in pycol_cols[:12]:
            if pd.notna(r.get(c)):
                preview.append({"metric": c, "value": r[c]})
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="scf_s2_back"):
            st.session_state["scf_step"] = 1
            st.rerun()
    with c2:
        if st.button("Continue →", type="primary", key="scf_s2_next"):
            st.session_state["scf_step"] = 3
            st.rerun()

elif step == 3:
    st.header("Step 3 — Metrics to boost & donors")

    anchor = st.session_state.get("scf_anchor", "")
    if not anchor:
        st.warning("Pick an anchor in Step 2.")
        st.stop()

    st.info(f"Anchor: **{anchor}** — new samples will use **this dataset's classes only**.")

    boost = st.multiselect(
        "Metrics to make harder",
        options=PYCOL_BOOST_METRICS,
        default=st.session_state.get("scf_boost_metrics", ["C1"]),
        format_func=lambda m: f"{m} ({metric_direction_label(PYCOL_METRICS[m].direction)})",
        key="scf_boost_pick",
    )
    st.session_state["scf_boost_metrics"] = boost

    if not boost:
        st.stop()

    summary_path = st.session_state.get("scf_summary_path", str(resolve_complexity_summary_path()))
    profiled = _cached_profile(
        _summary_mtime(Path(summary_path)),
        str(summary_path),
        float(st.session_state.get("scf_archetype_thr", 0.75)),
    )

    st.subheader("Top donors per metric (from corpus)")
    st.dataframe(exemplar_table(profiled, boost, top_k=3), use_container_width=True, hide_index=True)

    donor_mode = st.radio(
        "Donor selection",
        ["auto", "manual"],
        format_func=lambda x: "Auto (hardest dataset per metric)" if x == "auto" else "Manual",
        key="scf_donor_mode",
    )

    pmlb_names = list_downloaded_datasets(DEFAULT_OUTPUT_DIR)
    manual_donors: dict[str, str] = {}
    if donor_mode == "manual":
        for m in boost:
            picks = top_exemplars_for_metric(profiled, m, top_k=1)
            default_ds = picks[0].dataset_file.removesuffix(".csv") if picks else pmlb_names[0]
            opts = [n for n in pmlb_names if n != anchor] or pmlb_names
            idx = opts.index(default_ds) if default_ds in opts else 0
            manual_donors[m] = st.selectbox(f"Donor for {m}", opts, index=idx, key=f"scf_donor_{m}")
        st.session_state["scf_manual_donors"] = manual_donors
    else:
        st.session_state.pop("scf_manual_donors", None)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="scf_s3_back"):
            st.session_state["scf_step"] = 2
            st.rerun()
    with c2:
        if st.button("Continue →", type="primary", key="scf_s3_next"):
            st.session_state["scf_step"] = 4
            st.rerun()

elif step == 4:
    st.header("Step 4 — Augmentation settings")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("New samples per class", 10, 5000, 200, 10, key="scf_new_per_class")
    with c2:
        st.slider("Perturbation strength", 0.0, 0.9, 0.35, 0.05, key="scf_perturb")
    with c3:
        st.number_input("Random seed", 0, 999_999, 42, key="scf_seed")

    st.checkbox("Keep original anchor rows", value=True, key="scf_keep_original")
    st.number_input("Overlap noise", 0.0, 0.2, 0.02, 0.01, key="scf_overlap_noise")

    verify_default = st.session_state.get("scf_boost_metrics", ["C1"])
    st.multiselect(
        "Verify metrics (before vs after)",
        PYCOL_BOOST_METRICS,
        default=verify_default,
        key="scf_verify_metrics",
    )
    st.text_input("Output filename stem", value=f"{st.session_state.get('scf_anchor', 'anchor')}_harder", key="scf_output_stem")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="scf_s4_back"):
            st.session_state["scf_step"] = 3
            st.rerun()
    with c2:
        if st.button("Continue →", type="primary", key="scf_s4_next"):
            st.session_state["scf_step"] = 5
            st.rerun()

else:
    st.header("Step 5 — Generate & save")

    anchor = st.session_state.get("scf_anchor", "")
    boost = st.session_state.get("scf_boost_metrics", [])
    if not anchor or not boost:
        st.error("Complete steps 2–3 first.")
        st.stop()

    donor_mode = st.session_state.get("scf_donor_mode", "auto")
    manual = dict(st.session_state.get("scf_manual_donors", {})) if donor_mode == "manual" else {}

    config = AnchorAugmentConfig(
        anchor_dataset=anchor,
        boost_metrics=boost,
        donor_per_metric=manual,
        new_samples_per_class=int(st.session_state.get("scf_new_per_class", 200)),
        keep_original=bool(st.session_state.get("scf_keep_original", True)),
        perturbation_strength=float(st.session_state.get("scf_perturb", 0.35)),
        overlap_noise=float(st.session_state.get("scf_overlap_noise", 0.02)),
        random_seed=int(st.session_state.get("scf_seed", 42)),
        pmlb_dir=DEFAULT_OUTPUT_DIR,
        summary_path=Path(st.session_state.get("scf_summary_path", str(resolve_complexity_summary_path()))),
        verify_metrics=list(st.session_state.get("scf_verify_metrics", boost)),
        output_name=str(st.session_state.get("scf_output_stem", f"{anchor}_harder")),
    )

    st.json(
        {
            "anchor": config.anchor_dataset,
            "boost_metrics": config.boost_metrics,
            "new_samples_per_class": config.new_samples_per_class,
            "keep_original": config.keep_original,
            "perturbation_strength": config.perturbation_strength,
            "manual_donors": manual or None,
        }
    )

    col_g, col_b = st.columns([2, 1])
    with col_b:
        if st.button("← Back", key="scf_s5_back"):
            st.session_state["scf_step"] = 4
            st.rerun()
    with col_g:
        run = st.button("Generate harder dataset", type="primary", key="scf_generate")

    if run:
        with st.spinner("Generating samples in anchor classes…"):
            try:
                result = generate_augmented_dataset(config)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        st.session_state["scf_result_df"] = result.dataframe
        st.session_state["scf_result_meta"] = {
            "donors": [
                {
                    "metric": d.metric,
                    "donor": d.donor_name,
                    "hardness_rank": d.hardness_rank,
                    "overlap_intensity": d.overlap_intensity,
                }
                for d in result.donors
            ],
            "metadata": result.metadata,
            "anchor_metrics": result.anchor_metrics,
            "augmented_metrics": result.augmented_metrics,
        }

    result_df = st.session_state.get("scf_result_df")
    meta = st.session_state.get("scf_result_meta")

    if result_df is not None and not result_df.empty and meta:
        st.success(
            f"**{meta['metadata'].get('n_rows_total')}** rows "
            f"({meta['metadata'].get('n_rows_original')} original + "
            f"{meta['metadata'].get('n_rows_new')} new)"
        )

        if meta.get("donors"):
            st.dataframe(pd.DataFrame(meta["donors"]), use_container_width=True, hide_index=True)

        if meta.get("anchor_metrics"):
            st.subheader("Metrics: anchor → augmented")
            rows = []
            for m in sorted(set(meta["anchor_metrics"]) | set(meta.get("augmented_metrics", {}))):
                rows.append(
                    {
                        "metric": metric_display_name(m, "pycol"),
                        "anchor": meta["anchor_metrics"].get(m),
                        "augmented": meta.get("augmented_metrics", {}).get(m),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "Higher/lower meaning depends on metric direction (see Metric Reference). "
                "Goal: move toward **harder** on chosen boost metrics."
            )

        st.dataframe(result_df.head(12), use_container_width=True)
        stem = str(st.session_state.get("scf_output_stem", f"{anchor}_harder"))
        render_save_results_section(
            result_df,
            default_filename=f"{stem}.csv",
            key_prefix="scf_save",
            results_dir=DEFAULT_SYNTHETIC_DIR,
        )

        if st.button("Add to Dataset Comparison", key="scf_add_cmp"):
            if "comparison_datasets" not in st.session_state:
                st.session_state["comparison_datasets"] = []
            st.session_state["comparison_datasets"].append(
                {
                    "dataset_name": stem,
                    "df": result_df.copy(),
                    "label_col": config.label_col,
                    "source": f"Augmented from {anchor}",
                }
            )
            st.success(f"Added **{stem}** to Dataset Comparison.")
