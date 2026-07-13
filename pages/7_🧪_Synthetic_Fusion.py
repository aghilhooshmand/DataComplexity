from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from complexity_profiler import (
    exemplar_table,
    profile_summary,
    stress_source_datasets,
    top_exemplars_for_metric,
)
from metric_catalog import PYCOL_METRICS, metric_direction_label
from metric_ui import metric_display_name
from pmlb_io import DEFAULT_OUTPUT_DIR, list_downloaded_datasets
from results_export import render_save_results_section
from summary_dashboard import resolve_complexity_summary_path
from synthetic_fusion import SyntheticFusionConfig, generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_DIR = PROJECT_ROOT / "results" / "synthetic"

PYCOL_FUSION_METRICS = sorted(PYCOL_METRICS.keys())


@st.cache_data(show_spinner=False)
def _cached_profile(summary_mtime_ns: int, summary_path: str, threshold: float) -> pd.DataFrame:
    return profile_summary(summary_path=summary_path, threshold=threshold)


def _init_session() -> None:
    defaults = {
        "scf_step": 1,
        "scf_result_df": None,
        "scf_result_meta": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _summary_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


st.set_page_config(page_title="Synthetic Fusion", layout="wide")
st.title("🧪 Synthetic Complexity Fusion")
st.markdown(
    """
Build **hard synthetic datasets** by fusing patterns from real PMLB datasets ranked on PyCol metrics.
Workflow: **profile → pick metrics & sources → generate → verify → save**.

CLI equivalent: `python synthetic_fusion_cli.py --help`
"""
)

_init_session()

steps = [
    "1. Load complexity profile",
    "2. Mode & target metrics",
    "3. Source datasets",
    "4. Generation settings",
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
    st.caption("Uses `datasets_complexity_summary.csv` to rank datasets per metric (Phase 1).")

    try:
        default_summary = resolve_complexity_summary_path()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    summary_path = st.text_input(
        "Complexity summary CSV",
        value=str(default_summary),
        key="scf_summary_path",
    )
    archetype_threshold = st.slider(
        "Archetype hardness threshold (percentile)",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        step=0.05,
        key="scf_archetype_thr",
    )

    path = Path(summary_path)
    if not path.is_file():
        st.warning("Summary file not found. Run batch complexity first or point to a valid CSV.")
        st.stop()

    profiled = _cached_profile(_summary_mtime(path), str(path), float(archetype_threshold))
    st.success(f"Loaded **{len(profiled)}** datasets from `{path.name}`.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Datasets", len(profiled))
    with col_b:
        st.metric("With archetype tags", int((profiled["archetypes"] != "moderate").sum()))
    with col_c:
        hard_cols = [c for c in profiled.columns if c.startswith("hard_")]
        st.metric("Ranked metrics", len(hard_cols))

    preview_cols = ["dataset_file", "archetypes"] + hard_cols[:6]
    preview_cols = [c for c in preview_cols if c in profiled.columns]
    st.dataframe(profiled[preview_cols].head(20), use_container_width=True, hide_index=True)

    if st.button("Continue to metrics →", type="primary", key="scf_s1_next"):
        st.session_state["scf_step"] = 2
        st.rerun()

elif step == 2:
    st.header("Step 2 — Fusion mode & target metrics")

    mode = st.radio(
        "Fusion mode",
        options=["targeted", "stress"],
        format_func=lambda m: {
            "targeted": "Targeted — one exemplar per metric (feature-block fusion)",
            "stress": "Stress — combine several globally hard datasets",
        }[m],
        key="scf_mode",
    )

    default_metrics = st.session_state.get("scf_metrics", ["F1", "F2"])
    metrics = st.multiselect(
        "Target metrics",
        options=PYCOL_FUSION_METRICS,
        default=default_metrics,
        format_func=lambda m: f"{m} ({metric_direction_label(PYCOL_METRICS[m].direction)})",
        key="scf_metrics_pick",
    )
    st.session_state["scf_metrics"] = metrics

    if not metrics:
        st.info("Select at least one metric.")
        st.stop()

    if mode == "stress":
        st.session_state["scf_top_sources"] = st.number_input(
            "Number of stress sources (top hard datasets)",
            min_value=2,
            max_value=8,
            value=int(st.session_state.get("scf_top_sources", 4)),
            key="scf_top_sources_input",
        )

    st.markdown("#### Metric direction reminder")
    dir_rows = [
        {
            "metric": metric_display_name(m, "pycol"),
            "direction": metric_direction_label(PYCOL_METRICS[m].direction),
        }
        for m in metrics
    ]
    st.dataframe(pd.DataFrame(dir_rows), use_container_width=True, hide_index=True)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="scf_s2_back"):
            st.session_state["scf_step"] = 1
            st.rerun()
    with col_next:
        if st.button("Continue to sources →", type="primary", key="scf_s2_next"):
            st.session_state["scf_step"] = 3
            st.rerun()

elif step == 3:
    st.header("Step 3 — Source datasets")

    mode = st.session_state.get("scf_mode", "targeted")
    metrics = st.session_state.get("scf_metrics", [])
    summary_path = st.session_state.get("scf_summary_path", str(resolve_complexity_summary_path()))
    threshold = float(st.session_state.get("scf_archetype_thr", 0.75))

    path = Path(summary_path)
    if not path.is_file():
        st.error("Go back to Step 1 and load a valid summary CSV.")
        st.stop()

    profiled = _cached_profile(_summary_mtime(path), str(summary_path), threshold)
    pmlb_names = list_downloaded_datasets(DEFAULT_OUTPUT_DIR)

    st.subheader("Auto exemplars (top per metric)")
    ex_table = exemplar_table(profiled, metrics, top_k=3)
    if not ex_table.empty:
        st.dataframe(ex_table, use_container_width=True, hide_index=True)
    else:
        st.warning("No exemplars found for selected metrics.")

    pick_mode = st.radio(
        "Source selection",
        options=["auto", "manual"],
        format_func=lambda x: "Auto (top exemplar per metric)" if x == "auto" else "Manual override",
        key="scf_source_mode",
    )

    if pick_mode == "manual" and mode == "targeted":
        st.caption("Choose one PMLB dataset per metric (feature block).")
        manual_map: dict[str, str] = {}
        for m in metrics:
            picks = top_exemplars_for_metric(profiled, m, top_k=1)
            default_ds = picks[0].dataset_file.removesuffix(".csv") if picks else ""
            options = pmlb_names or ([default_ds] if default_ds else [])
            idx = options.index(default_ds) if default_ds in options else 0
            choice = st.selectbox(
                f"{m} source",
                options=options,
                index=idx,
                key=f"scf_manual_{m}",
            )
            manual_map[m] = choice
        st.session_state["scf_manual_map"] = manual_map
    elif pick_mode == "manual" and mode == "stress":
        default_stress = stress_source_datasets(
            profiled,
            metrics=metrics,
            top_n=int(st.session_state.get("scf_top_sources", 4)),
        )
        default_stems = [f.removesuffix(".csv") for f in default_stress]
        chosen = st.multiselect(
            "Stress source datasets",
            options=pmlb_names,
            default=[d for d in default_stems if d in pmlb_names],
            key="scf_stress_sources",
        )
        st.session_state["scf_stress_sources"] = chosen
    else:
        st.session_state.pop("scf_manual_map", None)
        st.session_state.pop("scf_stress_sources", None)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="scf_s3_back"):
            st.session_state["scf_step"] = 2
            st.rerun()
    with col_next:
        if st.button("Continue to settings →", type="primary", key="scf_s3_next"):
            st.session_state["scf_step"] = 4
            st.rerun()

elif step == 4:
    st.header("Step 4 — Generation settings")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Samples per class",
            min_value=50,
            max_value=5000,
            value=500,
            step=50,
            key="scf_samples",
        )
    with c2:
        st.number_input(
            "Coupling noise",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            key="scf_noise",
        )
    with c3:
        st.number_input("Random seed", min_value=0, max_value=999_999, value=42, key="scf_seed")

    verify_default = st.session_state.get("scf_metrics", ["F1", "F2"])
    st.multiselect(
        "Verify after generation (recompute PyCol)",
        options=PYCOL_FUSION_METRICS,
        default=verify_default,
        key="scf_verify_metrics",
    )
    st.text_input("Output filename stem", value="synthetic_fusion", key="scf_output_stem")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="scf_s4_back"):
            st.session_state["scf_step"] = 3
            st.rerun()
    with col_next:
        if st.button("Continue to generate →", type="primary", key="scf_s4_next"):
            st.session_state["scf_step"] = 5
            st.rerun()

else:
    st.header("Step 5 — Generate & save")

    mode = st.session_state.get("scf_mode", "targeted")
    metrics = st.session_state.get("scf_metrics", [])
    summary_path = Path(st.session_state.get("scf_summary_path", str(resolve_complexity_summary_path())))
    pick_mode = st.session_state.get("scf_source_mode", "auto")

    source_files: list[str] = []
    metric_map: dict[str, str] = {}
    if pick_mode == "manual":
        if mode == "targeted":
            metric_map = dict(st.session_state.get("scf_manual_map", {}))
        else:
            source_files = list(st.session_state.get("scf_stress_sources", []))

    config = SyntheticFusionConfig(
        mode=mode,
        target_metrics=metrics,
        source_files=source_files,
        metric_to_source=metric_map,
        samples_per_class=int(st.session_state.get("scf_samples", 500)),
        coupling_noise=float(st.session_state.get("scf_noise", 0.05)),
        random_seed=int(st.session_state.get("scf_seed", 42)),
        pmlb_dir=DEFAULT_OUTPUT_DIR,
        summary_path=summary_path,
        verify_metrics=list(st.session_state.get("scf_verify_metrics", [])),
        output_name=str(st.session_state.get("scf_output_stem", "synthetic_fusion")),
    )

    if mode == "stress" and pick_mode == "auto":
        profiled = profile_summary(summary_path=summary_path)
        config.source_files = [
            f.removesuffix(".csv")
            for f in stress_source_datasets(
                profiled,
                metrics=metrics,
                top_n=int(st.session_state.get("scf_top_sources", 4)),
            )
        ]

    st.markdown("#### Configuration summary")
    st.json(
        {
            "mode": config.mode,
            "metrics": config.target_metrics,
            "samples_per_class": config.samples_per_class,
            "coupling_noise": config.coupling_noise,
            "seed": config.random_seed,
            "verify": config.verify_metrics,
            "manual_sources": metric_map or None,
            "stress_sources": config.source_files or None,
        }
    )

    col_gen, col_back = st.columns([2, 1])
    with col_back:
        if st.button("← Back", key="scf_s5_back"):
            st.session_state["scf_step"] = 4
            st.rerun()

    with col_gen:
        run = st.button("Generate synthetic dataset", type="primary", key="scf_generate")

    if run:
        with st.spinner("Fusing patterns and sampling…"):
            try:
                result = generate_synthetic_dataset(config)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()

        st.session_state["scf_result_df"] = result.dataframe
        st.session_state["scf_result_meta"] = {
            "metric_sources": result.metric_sources,
            "metadata": result.metadata,
            "verified_metrics": result.verified_metrics,
            "patterns": [
                {
                    "source": p.source_name,
                    "file": p.source_file,
                    "metric_tag": p.metric_tag,
                    "n_features": p.n_features,
                }
                for p in result.patterns
            ],
        }
        st.success(
            f"Generated **{len(result.dataframe)}** rows, "
            f"**{result.metadata.get('n_features')}** features."
        )

    result_df: pd.DataFrame | None = st.session_state.get("scf_result_df")
    meta = st.session_state.get("scf_result_meta")

    if result_df is not None and not result_df.empty:
        if meta:
            st.subheader("Fusion metadata")
            if meta.get("metric_sources"):
                st.caption("Metric → source file")
                st.json(meta["metric_sources"])
            if meta.get("patterns"):
                st.dataframe(pd.DataFrame(meta["patterns"]), use_container_width=True, hide_index=True)
            if meta.get("verified_metrics"):
                st.subheader("Verified PyCol metrics (synthetic set)")
                vdf = pd.DataFrame(
                    [
                        {
                            "metric": metric_display_name(m, "pycol"),
                            "value": v,
                        }
                        for m, v in meta["verified_metrics"].items()
                    ]
                )
                st.dataframe(vdf, use_container_width=True, hide_index=True)

        st.subheader("Preview")
        st.dataframe(result_df.head(15), use_container_width=True)

        stem = str(st.session_state.get("scf_output_stem", "synthetic_fusion"))
        render_save_results_section(
            result_df,
            default_filename=f"{stem}.csv",
            key_prefix="scf_save",
            results_dir=DEFAULT_SYNTHETIC_DIR,
        )

        st.caption(f"Default synthetic folder: `{DEFAULT_SYNTHETIC_DIR}`")

        if meta and st.button("Add to Dataset Comparison", key="scf_add_comparison"):
            label_col = config.label_col
            if "comparison_datasets" not in st.session_state:
                st.session_state["comparison_datasets"] = []
            st.session_state["comparison_datasets"].append(
                {
                    "dataset_name": stem,
                    "df": result_df.copy(),
                    "label_col": label_col,
                    "source": "Synthetic Fusion",
                }
            )
            st.success(f"Added **{stem}** to Dataset Comparison.")
