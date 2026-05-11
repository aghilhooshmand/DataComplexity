#!/usr/bin/env python3
"""
Run parallel_complexity_cli.py once per row in a TSV manifest (sequential jobs;
each job still uses --n-jobs workers internally).

Manifest columns (tab-separated, header row required):
  source         uci | openml | csv
  ref            UCI/OpenML URL or id, or path to a CSV file when source=csv
  label_column   target column name in the dataframe seen by the CLI
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path


def _default_n_jobs() -> int:
    try:
        return max(1, (mp.cpu_count() or 4) // 2)
    except NotImplementedError:
        return 4


def read_manifest(path: Path) -> list[dict[str, str]]:
    from io import StringIO

    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("source\t") or low.startswith("source,"):
            start = i
            break
    else:
        raise ValueError(f"No header row starting with 'source' in manifest: {path}")
    header_line = lines[start]
    delim = "\t" if "\t" in header_line else ","
    rows: list[dict[str, str]] = []
    buf = StringIO("\n".join(lines[start:]))
    reader = csv.DictReader(buf, delimiter=delim)
    for raw in reader:
        row = {k.strip(): (v or "").strip() for k, v in raw.items()}
        ref = row.get("ref", "").strip() or row.get("path", "").strip()
        if not ref or ref.startswith("#"):
            continue
        source = row.get("source", "").strip().lower()
        label = row.get("label_column", "").strip()
        if not source or not label:
            continue
        rows.append({"source": source, "ref": ref, "label_column": label})
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    cli = root / "parallel_complexity_cli.py"

    parser = argparse.ArgumentParser(description="Batch-run parallel_complexity_cli.py from a TSV manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "batch_manifest.tsv",
        help="Tab-separated manifest (default: batch_manifest.tsv next to this script).",
    )
    parser.add_argument("--library", default="both", choices=["pycol", "pymfe", "both"])
    parser.add_argument("--metrics", default="all", help="Used when library is pycol or pymfe only.")
    parser.add_argument("--pycol-metrics", default="all")
    parser.add_argument("--pymfe-metrics", default="all")
    parser.add_argument("--n-jobs", type=int, default=_default_n_jobs())
    parser.add_argument(
        "--missing-values",
        default="impute_median",
        choices=["drop_rows", "fill_zero", "impute_median", "impute_mean"],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "results" / "batch_parallel_complexity.csv",
        help="Single CSV; each dataset upserts one row (same as CLI).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run.")
    parser.add_argument("--no-progress", action="store_true", help="Pass --no-progress to each CLI run.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going if one dataset fails.")
    args = parser.parse_args()

    if not cli.is_file():
        sys.exit(f"CLI not found: {cli}")

    rows = read_manifest(args.manifest)
    if not rows:
        sys.exit(f"No data rows in manifest: {args.manifest}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {args.manifest}  ({len(rows)} dataset(s))", flush=True)
    print(f"Output CSV: {args.output_csv}", flush=True)

    failed: list[tuple[int, str, int]] = []
    for i, row in enumerate(rows, start=1):
        cmd = [
            sys.executable,
            str(cli),
            "--source",
            row["source"],
            "--ref",
            row["ref"],
            "--label-column",
            row["label_column"],
            "--library",
            args.library,
            "--n-jobs",
            str(args.n_jobs),
            "--missing-values",
            args.missing_values,
            "--output-csv",
            str(args.output_csv),
        ]
        if args.library == "pycol":
            cmd.extend(["--metrics", args.metrics])
        elif args.library == "pymfe":
            cmd.extend(["--metrics", args.metrics])
        else:
            cmd.extend(["--pycol-metrics", args.pycol_metrics, "--pymfe-metrics", args.pymfe_metrics])
        if args.no_progress:
            cmd.append("--no-progress")

        print(f"\n========== [{i}/{len(rows)}] {row['source']}  {row['ref'][:80]}{'…' if len(row['ref']) > 80 else ''} ==========", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue

        r = subprocess.run(cmd, cwd=str(root))
        if r.returncode != 0:
            failed.append((i, row["ref"], r.returncode))
            if not args.continue_on_error:
                sys.exit(r.returncode)

    if failed:
        print("\nCompleted with errors:", flush=True)
        for item in failed:
            print(f"  row {item[0]} code {item[2]}: {item[1]}", flush=True)
        sys.exit(1)
    print("\nAll batch jobs finished OK.", flush=True)


if __name__ == "__main__":
    main()
