#!/usr/bin/env python3
"""Download PMLB classification datasets (filtered by summary TSV) to pmlb_DS/."""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from pmlb_io import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SUMMARY_TSV,
    append_manifest_row,
    csv_path_for_dataset,
    encode_pmlb_dataframe,
    fetch_dataset_metadata_files,
    load_summary_table,
    metadata_dir_for_dataset,
    output_paths,
    select_benchmark_datasets,
    write_download_status,
)


def download_one(
    dataset_name: str,
    *,
    output_dir: Path,
    fetch_metadata: bool,
    missing_values: str,
    local_cache_dir: Path | None,
) -> dict:
    from pmlb import fetch_data

    t0 = time.perf_counter()
    kwargs: dict = {"dropna": True}
    if local_cache_dir is not None:
        kwargs["local_cache_dir"] = str(local_cache_dir)
    raw = fetch_data(dataset_name, **kwargs)
    if "target" not in raw.columns:
        raise ValueError("PMLB dataset has no 'target' column.")
    encoded = encode_pmlb_dataframe(raw, "target", missing_values=missing_values)
    csv_path = csv_path_for_dataset(output_dir, dataset_name)
    encoded.to_csv(csv_path, index=False)
    meta_saved: dict[str, bool] = {}
    if fetch_metadata:
        meta_saved = fetch_dataset_metadata_files(
            dataset_name, metadata_dir_for_dataset(output_dir, dataset_name)
        )
    elapsed = time.perf_counter() - t0
    return {
        "dataset": dataset_name,
        "status": "ok",
        "csv_file": csv_path.name,
        "n_rows_raw": int(raw.shape[0]),
        "n_rows_encoded": int(encoded.shape[0]),
        "n_columns_encoded": int(encoded.shape[1]),
        "n_features": int(encoded.shape[1] - 1),
        "n_classes": int(encoded["target"].nunique()),
        "readme_saved": meta_saved.get("README.md", False),
        "metadata_yaml_saved": meta_saved.get("metadata.yaml", False),
        "seconds": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download filtered PMLB classification datasets.")
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=DEFAULT_SUMMARY_TSV,
        help="PMLB all_summary_stats TSV (local copy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder (CSVs + metadata + manifest).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=21000,
        help="Keep datasets with n_instances strictly below this value.",
    )
    parser.add_argument(
        "--missing-values",
        default="impute_median",
        choices=["drop_rows", "fill_zero", "impute_median", "impute_mean"],
        help="Encoding strategy passed to prepare_xy when building CSVs.",
    )
    parser.add_argument(
        "--fetch-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download README.md and metadata.yaml from penn-ml-benchmarks.",
    )
    parser.add_argument(
        "--pmlb-cache-dir",
        type=Path,
        default=None,
        help="Optional local cache for raw .tsv.gz downloads.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip datasets whose CSV already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Download at most N datasets (0 = all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset names (must still pass filters unless --force-names).",
    )
    parser.add_argument(
        "--force-names",
        action="store_true",
        help="With --datasets, skip summary/PMLB list filtering.",
    )
    args = parser.parse_args(argv)

    try:
        from pmlb import classification_dataset_names
    except ImportError:
        print("Install pmlb: pip install pmlb", file=sys.stderr)
        return 1

    summary = load_summary_table(args.summary_tsv)
    pmlb_set = set(classification_dataset_names)
    selected = select_benchmark_datasets(
        summary,
        max_instances=args.max_instances,
        task="classification",
        pmlb_names=pmlb_set,
    )

    if args.datasets:
        if args.force_names:
            names = list(args.datasets)
        else:
            allowed = set(selected["dataset"].astype(str))
            names = [n for n in args.datasets if n in allowed]
    else:
        names = selected["dataset"].astype(str).tolist()

    if args.limit and args.limit > 0:
        names = names[: args.limit]

    paths = output_paths(args.output_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["metadata"].mkdir(parents=True, exist_ok=True)

    print(
        f"PMLB download: {len(names)} dataset(s) → {paths['root']} "
        f"(classification, n_instances < {args.max_instances})"
    )

    ok_rows: list[dict] = []
    failed: list[dict] = []
    skipped = 0

    for name in tqdm(names, desc="PMLB datasets"):
        csv_path = csv_path_for_dataset(paths["root"], name)
        if args.skip_existing and csv_path.is_file():
            skipped += 1
            continue
        try:
            row = download_one(
                name,
                output_dir=paths["root"],
                fetch_metadata=args.fetch_metadata,
                missing_values=args.missing_values,
                local_cache_dir=args.pmlb_cache_dir,
            )
            summary_row = summary.loc[summary["dataset"].astype(str) == name]
            if not summary_row.empty:
                for col in summary_row.columns:
                    if col not in row:
                        val = summary_row.iloc[0][col]
                        row[col] = val.item() if hasattr(val, "item") else val
            append_manifest_row(paths["manifest"], row)
            ok_rows.append(row)
        except Exception as exc:
            failed.append(
                {
                    "dataset": name,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    status = {
        "summary_tsv": str(args.summary_tsv),
        "output_dir": str(paths["root"]),
        "max_instances": args.max_instances,
        "requested": len(names),
        "downloaded_ok": len(ok_rows),
        "skipped_existing": skipped,
        "failed": len(failed),
        "failures": failed,
    }
    write_download_status(paths["status"], status)

    print(
        f"Done: {len(ok_rows)} new, {skipped} skipped, {len(failed)} failed. "
        f"Manifest: {paths['manifest']}"
    )
    if failed:
        for item in failed[:5]:
            print(f"  FAIL {item['dataset']}: {item['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
