#!/usr/bin/env python3
"""CLI: augment one anchor dataset with harder samples in its own classes (donor-guided overlap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from complexity_profiler import exemplar_table, profile_summary, pycol_key_to_column
from pmlb_io import DEFAULT_OUTPUT_DIR
from summary_dashboard import resolve_complexity_summary_path
from synthetic_fusion import AnchorAugmentConfig, generate_augmented_dataset, save_augmented_dataset

CLI_VERSION = "2.0.0"


def _parse_csv_list(text: str | None) -> list[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


def _parse_metric_sources(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in _parse_csv_list(text):
        if "=" not in part:
            continue
        metric, ds = part.split("=", 1)
        out[metric.strip()] = ds.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Augment one anchor PMLB dataset: add harder samples in its native classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List who is hardest on C1
  python synthetic_fusion_cli.py --list-exemplars --metrics C1 --top-k 5

  # DS1=ring: add samples so ring gets harder on C1 (donor auto-picked)
  python synthetic_fusion_cli.py --anchor ring --boost-metrics C1 \\
    --new-samples-per-class 200 --output results/synthetic/ring_harder_c1.csv \\
    --verify-metrics F1,C1,N3

  # Manual donor for C1
  python synthetic_fusion_cli.py --anchor ring --boost-metrics C1 \\
    --donor-sources C1=analcatdata_dmft \\
    --perturbation-strength 0.4 --output results/synthetic/ring_harder.csv
        """,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {CLI_VERSION}")

    p.add_argument(
        "--anchor",
        default="",
        help="Anchor dataset stem to make harder (e.g. ring). Required for augmentation.",
    )
    p.add_argument(
        "--boost-metrics",
        default="C1",
        help="Comma-separated metrics to increase hardness on (e.g. C1,N3).",
    )
    p.add_argument(
        "--donor-sources",
        default="",
        help="Optional metric=donor map, e.g. C1=analcatdata_dmft (else top exemplar per metric).",
    )
    p.add_argument(
        "--new-samples-per-class",
        type=int,
        default=200,
        help="New synthetic rows per anchor class (default 200).",
    )
    p.add_argument(
        "--no-keep-original",
        action="store_true",
        help="Output only new samples (default: keep original anchor rows).",
    )
    p.add_argument(
        "--perturbation-strength",
        type=float,
        default=0.35,
        help="Base push toward other-class centroids, 0–1 (default 0.35).",
    )
    p.add_argument(
        "--overlap-noise",
        type=float,
        default=0.02,
        help="Gaussian noise on perturbed samples (default 0.02).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--pmlb-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--verify-metrics", default="", help="Metrics to compare before/after.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--write-metadata", type=Path, default=None)
    p.add_argument("--list-exemplars", action="store_true")
    p.add_argument("--metrics", default="", help="With --list-exemplars: metrics to list.")
    p.add_argument("--top-k", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    summary_path = args.summary or resolve_complexity_summary_path()
    profiled = profile_summary(summary_path=summary_path)

    list_metrics = _parse_csv_list(args.metrics) or _parse_csv_list(args.boost_metrics)
    if args.list_exemplars:
        if not list_metrics:
            print("error: --metrics or --boost-metrics required with --list-exemplars", file=sys.stderr)
            return 2
        table = exemplar_table(profiled, list_metrics, top_k=int(args.top_k))
        if table.empty:
            print("No exemplars found.")
            return 1
        print(table.to_string(index=False))
        return 0

    if not str(args.anchor).strip():
        print("error: --anchor is required (e.g. --anchor ring)", file=sys.stderr)
        return 2
    if args.output is None:
        print("error: --output is required", file=sys.stderr)
        return 2

    boost = _parse_csv_list(args.boost_metrics)
    if not boost:
        print("error: --boost-metrics is required", file=sys.stderr)
        return 2

    config = AnchorAugmentConfig(
        anchor_dataset=str(args.anchor).strip(),
        boost_metrics=boost,
        donor_per_metric=_parse_metric_sources(args.donor_sources),
        new_samples_per_class=int(args.new_samples_per_class),
        keep_original=not args.no_keep_original,
        perturbation_strength=float(args.perturbation_strength),
        overlap_noise=float(args.overlap_noise),
        random_seed=int(args.seed),
        pmlb_dir=Path(args.pmlb_dir),
        summary_path=Path(summary_path),
        verify_metrics=_parse_csv_list(args.verify_metrics) or boost,
        output_name=Path(args.output).stem,
    )

    print(
        f"SCF {CLI_VERSION} — anchor={config.anchor_dataset} boost={boost}",
        file=sys.stderr,
    )

    try:
        result = generate_augmented_dataset(config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = save_augmented_dataset(result, args.output)
    print(f"Wrote: {out_path} ({len(result.dataframe)} rows)", file=sys.stderr)
    print(f"  alpha={result.metadata.get('perturbation_alpha', 0):.3f}", file=sys.stderr)

    for d in result.donors:
        print(
            f"  donor {d.metric}: {d.donor_name} "
            f"(hardness={d.hardness_rank:.2f}, overlap={d.overlap_intensity:.2f})",
            file=sys.stderr,
        )

    if result.anchor_metrics or result.augmented_metrics:
        print("  metrics (anchor → augmented):", file=sys.stderr)
        for m in sorted(set(result.anchor_metrics) | set(result.augmented_metrics)):
            a = result.anchor_metrics.get(m)
            b = result.augmented_metrics.get(m)
            a_s = f"{a:.6g}" if a is not None else "—"
            b_s = f"{b:.6g}" if b is not None else "—"
            print(f"    {pycol_key_to_column(m)}: {a_s} → {b_s}", file=sys.stderr)

    if args.write_metadata:
        meta_path = Path(args.write_metadata)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cli_version": CLI_VERSION,
            "output_csv": str(out_path),
            "donors": [
                {
                    "metric": d.metric,
                    "donor": d.donor_name,
                    "file": d.donor_file,
                    "hardness_rank": d.hardness_rank,
                    "overlap_intensity": d.overlap_intensity,
                }
                for d in result.donors
            ],
            "anchor_metrics": result.anchor_metrics,
            "augmented_metrics": result.augmented_metrics,
            **result.metadata,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {meta_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
