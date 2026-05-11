#!/usr/bin/env bash
# Batch-run parallel_complexity_cli.py for each row in batch_manifest.tsv (one dataset after another).
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_batch_parallel.py "$@"
