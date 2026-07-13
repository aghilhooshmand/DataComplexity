#!/usr/bin/env python3
"""CLI for Synthetic Complexity Fusion (SCF): build hard synthetic datasets from PMLB exemplars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from complexity_profiler import exemplar_table, profile_summary, pycol_key_to_column
from pmlb_io import DEFAULT_OUTPUT_DIR
from summary_dashboard import resolve_complexity_summary_path
from synthetic_fusion import SyntheticFusionConfig, generate_synthetic_dataset, save_synthetic_dataset

CLI_VERSION = "1.0.0"


def _parse_csv_list(text: str | None) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _parse_metric_sources(text: str | None) -> dict[str, str]:
    """Parse F1=ring,F2=twonorm style mapping."""
    out: dict[str, str] = {}
    for part in _parse_csv_list(text):
        if "=" not in part:
            continue
        metric, ds = part.split("=", 1)
        out[metric.strip()] = ds.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthetic Complexity Fusion (SCF): fuse hardness patterns from real datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Targeted: auto-pick exemplars for F1 and F2, verify after generation
  python synthetic_fusion_cli.py --mode targeted --metrics F1,F2 \\
    --output results/synthetic_f1_f2.csv --verify-metrics F1,F2,N3

  # Targeted: manual dataset per metric
  python synthetic_fusion_cli.py --mode targeted --metrics F1,F2 \\
    --metric-sources F1=ring,F2=twonorm --output results/synthetic_manual.csv

  # Stress: top hard datasets across many metrics
  python synthetic_fusion_cli.py --mode stress \\
    --metrics F1,F2,N3,N1,borderline --top-sources 4 \\
    --output results/synthetic_stress.csv --verify-metrics F1,F2,N3
        """,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")

    p.add_argument(
        "--mode",
        choices=("targeted", "stress"),
        default="targeted",
        help="targeted = one exemplar per metric (feature-block fusion); stress = top hard datasets combined.",
    )
    p.add_argument(
        "--metrics",
        default="F1,F2",
        help="Comma-separated PyCol metric names (without pycol_ prefix).",
    )
    p.add_argument(
        "--datasets",
        default="",
        help="Optional comma list of PMLB dataset stems. targeted: one per metric if counts match; else auto exemplars.",
    )
    p.add_argument(
        "--metric-sources",
        default="",
        help="Manual metric=dataset map, e.g. F1=ring,F2=twonorm (overrides --datasets).",
    )
    p.add_argument(
        "--top-sources",
        type=int,
        default=4,
        help="stress mode: number of hardest datasets to fuse (default 4).",
    )
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=500,
        help="Synthetic rows per class (default 500 → 1000 rows binary).",
    )
    p.add_argument(
        "--coupling-noise",
        type=float,
        default=0.05,
        help="Gaussian noise added across fused blocks (default 0.05).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--pmlb-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PMLB CSV folder (default: {DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Complexity summary CSV for exemplar ranking (default: results/ or pmlb_DS/).",
    )
    p.add_argument(
        "--verify-metrics",
        default="",
        help="Comma-separated metrics to recompute on the synthetic set after generation.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for synthetic dataset (required unless --list-exemplars).",
    )
    p.add_argument(
        "--write-metadata",
        type=Path,
        default=None,
        help="Optional JSON path for fusion metadata (sources, verified metrics).",
    )
    p.add_argument(
        "--list-exemplars",
        action="store_true",
        help="Print top exemplars per metric from the summary and exit (no synthesis).",
    )
    p.add_argument("--top-k", type=int, default=5, help="Rows per metric for --list-exemplars.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = _parse_csv_list(args.metrics)
    if not metrics:
        print("error: --metrics is required (e.g. F1,F2)", file=sys.stderr)
        return 2

    summary_path = args.summary
    if summary_path is None:
        summary_path = resolve_complexity_summary_path()

    profiled = profile_summary(summary_path=summary_path)

    if args.list_exemplars:
        table = exemplar_table(profiled, metrics, top_k=int(args.top_k))
        if table.empty:
            print("No exemplars found. Check summary CSV and metric names.")
            return 1
        print(table.to_string(index=False))
        return 0

    if args.output is None:
        print("error: --output is required for synthesis (omit only with --list-exemplars)", file=sys.stderr)
        return 2

    datasets = _parse_csv_list(args.datasets)
    metric_sources = _parse_metric_sources(args.metric_sources)

    config = SyntheticFusionConfig(
        mode=args.mode,
        target_metrics=metrics,
        source_files=datasets,
        metric_to_source=metric_sources,
        samples_per_class=int(args.samples_per_class),
        coupling_noise=float(args.coupling_noise),
        random_seed=int(args.seed),
        pmlb_dir=Path(args.pmlb_dir),
        summary_path=Path(summary_path),
        verify_metrics=_parse_csv_list(args.verify_metrics),
        output_name=Path(args.output).stem,
    )

    if args.mode == "stress" and not datasets:
        from complexity_profiler import stress_source_datasets

        config.source_files = stress_source_datasets(
            profiled, metrics=metrics, top_n=int(args.top_sources)
        )

    print(f"SCF {CLI_VERSION} — mode={config.mode} metrics={metrics}", file=sys.stderr)
    if metric_sources:
        print(f"  metric sources: {metric_sources}", file=sys.stderr)
    elif datasets:
        print(f"  datasets: {datasets}", file=sys.stderr)

    try:
        result = generate_synthetic_dataset(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = save_synthetic_dataset(result, args.output)
    print(f"Wrote synthetic dataset: {out_path} ({len(result.dataframe)} rows)", file=sys.stderr)
    print(f"  sources: {', '.join(result.metadata.get('source_files', []))}", file=sys.stderr)

    if result.metric_sources:
        print("  metric → source:", file=sys.stderr)
        for m, f in result.metric_sources.items():
            print(f"    {m} → {f}", file=sys.stderr)

    if result.verified_metrics:
        print("  verified metrics:", file=sys.stderr)
        for m, v in result.verified_metrics.items():
            print(f"    {pycol_key_to_column(m)} = {v:.6g}", file=sys.stderr)

    if args.write_metadata:
        meta_path = Path(args.write_metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cli_version": CLI_VERSION,
            "output_csv": str(out_path),
            "metric_sources": result.metric_sources,
            "patterns": [
                {
                    "source": p.source_name,
                    "file": p.source_file,
                    "metric_tag": p.metric_tag,
                    "n_features": p.n_features,
                }
                for p in result.patterns
            ],
            **result.metadata,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {meta_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
