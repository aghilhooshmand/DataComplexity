from __future__ import annotations

import streamlit as st

from complexity_core import METRIC_COST_HEURISTIC_CAPTION, available_metrics_by_library, get_cheap_expensive_pools


def render_cost_tier_metrics(library: str, *, key_prefix: str) -> list[str]:
    """
    When the user does not choose \"all metrics\", pick a cost mode and optionally
    subset within the cheap and/or expensive pools. Returns base metric names for ``library``.
    """
    cheap_pool, exp_pool = get_cheap_expensive_pools(library)
    if not cheap_pool and library == "pycol":
        st.error("Internal error: empty cheap pool for pycol.")
        return []

    mode = st.radio(
        f"**{library}** — cost pool mode",
        options=["custom_both", "cheap_only", "expensive_only"],
        format_func=lambda x: {
            "custom_both": "Cheap + expensive (customize each pool)",
            "cheap_only": "Cheap pool only",
            "expensive_only": "Expensive pool only",
        }[x],
        key=f"{key_prefix}_{library}_tier_mode",
        horizontal=False,
    )

    if mode == "cheap_only":
        return st.multiselect(
            f"{library}: metrics in **cheap** pool",
            options=cheap_pool,
            default=cheap_pool,
            key=f"{key_prefix}_{library}_cheap_only",
        )

    if mode == "expensive_only":
        if not exp_pool:
            st.warning("No metrics in the expensive pool for this library/runtime.")
            return []
        return st.multiselect(
            f"{library}: metrics in **expensive** pool",
            options=exp_pool,
            default=exp_pool,
            key=f"{key_prefix}_{library}_exp_only",
        )

    st.caption("Selections from both pools are **combined** (union) for this run.")
    c = st.multiselect(
        f"{library}: **cheap** pool",
        options=cheap_pool,
        default=cheap_pool,
        key=f"{key_prefix}_{library}_cheap_pick",
    )
    e = st.multiselect(
        f"{library}: **expensive** pool",
        options=exp_pool,
        default=exp_pool,
        key=f"{key_prefix}_{library}_exp_pick",
    )
    return sorted(set(c) | set(e))


def render_metric_selection_block(
    selected_libraries: list[str],
    *,
    key_prefix: str,
    use_all_label: str = "Use all metrics",
) -> dict[str, list[str]]:
    """
    Returns ``selected_by_library`` mapping each selected library to its metric name list.
    """
    selected_by_library: dict[str, list[str]] = {}
    if not selected_libraries:
        return selected_by_library

    use_all = st.checkbox(use_all_label, value=True, key=f"{key_prefix}_use_all")

    with st.expander("Cost tiers (cheap vs expensive) — how we group metrics", expanded=False):
        st.markdown(METRIC_COST_HEURISTIC_CAPTION)

    for lib in selected_libraries:
        all_metrics = available_metrics_by_library(lib)
        st.markdown(f"#### {lib.upper()}")
        if use_all:
            selected_by_library[lib] = all_metrics
            st.caption(f"All **{len(all_metrics)}** metrics selected.")
        else:
            selected_by_library[lib] = render_cost_tier_metrics(lib, key_prefix=key_prefix)

    return selected_by_library
