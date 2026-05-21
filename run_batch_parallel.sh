#!/usr/bin/env bash
#
# Run parallel_complexity_cli.py once per dataset, in order.
# Edit the variables below (defaults + dataset list), then:  chmod +x run_batch_parallel.sh && ./run_batch_parallel.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/parallel_complexity_cli.py"
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Shared CLI arguments (edit here)
# Tuned for: many-core server (e.g. 80+ CPUs, ~125 GiB RAM), full samples, cheap + auto HEOM tier.
# ---------------------------------------------------------------------------
LIBRARY="pycol"                    # pycol | pymfe | both
# Worker cap for HEOM rows (--pycol-parallel-heom) and PyCol metric pool (--pycol-parallel-metrics).
# CLI uses min(N_JOBS, #metrics, CPU count). 24–32 is enough on an 82-core box; avoid N_JOBS=82.
N_JOBS="24"
MISSING_VALUES="impute_median"   # drop_rows | fill_zero | impute_median | impute_mean
OUTPUT_CSV="${SCRIPT_DIR}/results/batch_parallel_complexity.csv"

# Default max rows per dataset (0 = all rows after cleaning).
COMPLEXITY_MAX_ROWS="0"

# Parallel PyCol metrics (one process per metric). Each distance metric rebuilds the full n×n
# matrix → on adult/CDC with build this can exceed 125 GiB. Keep "0" for full-sample batch;
# set "1" only for small n (e.g. breast) or with COMPLEXITY_MAX_ROWS capped.
PYCOL_PARALLEL_METRICS="0"

# PyCol preset (when LIBRARY is pycol, or the PyCol side when LIBRARY is both).
# Only PYCOL_METRICS_ARG is passed to the CLI unless you set it to "custom".
#
#   cheap_minimal — overlap/purity only; WHY: no row×row distance table (fastest, lowest RAM)
#   cheap         — all PyCol except T1, NSG, ICSV (incl. N4, kDN, …); WHY: at most ONE matrix
#   expensive_core — T1, NSG, ICSV only; WHY: need unnormalized HEOM → TWO matrices
#   expensive     — same as expensive_core
#   all           — full catalog (cheap + expensive_core rules combined)
#   custom        — use PYCOL_CUSTOM_METRICS below (comma list), e.g. N1,N3,F1,F1v
#   N1,N3,F1      — bare comma list also works (PYCOL_CUSTOM_METRICS still ignored)
#
# Active setting below: "cheap" → ~26 metrics, HEOM tier from PYCOL_DISTANCE_MATRIX=auto → one matrix max.
PYCOL_METRICS_ARG="custom"
# Ignored while PYCOL_METRICS_ARG is not "custom" (example list for when you switch to custom):
PYCOL_CUSTOM_METRICS="F1,F2,F3,F4,F1v,input_noise,purity,N2,N3,C1,C2"

# PyCol HEOM RAM tier (only when LIBRARY is pycol or both):
#   auto  — from preset: cheap_minimal→skip, cheap→dist, expensive*→both; custom→infer
#   skip | dist | both — force tier (dist = one matrix ~½ RAM of both)
PYCOL_DISTANCE_MATRIX="auto"

# Parallel HEOM row workers (when tier is dist or both):
PYCOL_PARALLEL_HEOM="1"

# Per-dataset row cap (optional 4th field in DATASETS): source|ref|label|max_rows
# UCI 891 (CDC) ~253k rows: full n×n build needs ~1 TiB RAM — not feasible on 125 GiB.
# 75000 rows ≈ ~90 GiB for two distance matrices alone; leave headroom for workers.

# When LIBRARY is "both":
PYMFE_METRICS="all"

# When LIBRARY is "pymfe" only:
METRICS_SINGLE="all"

# NO_PROGRESS: whether each CLI run adds flag --no-progress
#   "0" = show normal CLI progress (phases + tqdm on stderr) when stderr is a terminal
#   "1" = silent runs (no progress bar / no step lines) — use for logs, cron, or non-TTY pipes
NO_PROGRESS="0"

# CONTINUE_ON_ERROR: after a failed dataset
#   "0" = stop the whole batch immediately
#   "1" = print the error and continue with the next dataset
# Override on the command line, e.g. CONTINUE_ON_ERROR=0 DRY_RUN=1 ./run_batch_parallel.sh
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# DRY_RUN:
#   "0" = really run parallel_complexity_cli.py for each line
#   "1" = only print the shell-quoted command that would be run (no execution)
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Dataset list: one entry per line inside the array.
# Format:   source|ref|label_column|[max_rows]
#   source = uci | openml | csv
#   ref     = UCI/OpenML URL or id, OR absolute path to CSV when source=csv
#   label_column = target column name (for uci/openml from this CLI, use "target")
#   max_rows (optional) = override COMPLEXITY_MAX_ROWS for this line only (e.g. CDC on 125 GiB)
# Use | as separator (do not use | inside URLs).
# ---------------------------------------------------------------------------
DATASETS=(
  "uci|https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic|target"
  "uci|https://archive.ics.uci.edu/dataset/2/adult|target"
  "uci|https://archive.ics.uci.edu/dataset/222/bank+marketing|target"
  "uci|https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping|target"
  "uci|https://archive.ics.uci.edu/dataset/186/wine+quality|target"
  "uci|https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators|target"
  "uci|https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset|target"
  "uci|https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume|target"
  # Example local CSV:
  # "csv|/absolute/path/to/data.csv|class"
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "${OUTPUT_CSV}")"

if [[ "${LIBRARY}" == "pycol" || "${LIBRARY}" == "both" ]]; then
  if [[ "${PYCOL_METRICS_ARG}" == "custom" ]] && [[ -z "${PYCOL_CUSTOM_METRICS// /}" ]]; then
    echo "PYCOL_METRICS_ARG is 'custom' but PYCOL_CUSTOM_METRICS is empty. Example: PYCOL_CUSTOM_METRICS='N1,N3,F1,F1v'" >&2
    exit 1
  fi
  case "${PYCOL_DISTANCE_MATRIX}" in
    auto|AUTO) PYCOL_DISTANCE_MATRIX="auto" ;;
    skip|SKIP) PYCOL_DISTANCE_MATRIX="skip" ;;
    dist|DIST|build|BUILD) PYCOL_DISTANCE_MATRIX="dist" ;;
    both|BOTH|unnorm|UNNORM) PYCOL_DISTANCE_MATRIX="both" ;;
    *)
      echo "PYCOL_DISTANCE_MATRIX must be auto, skip, dist, or both; got: ${PYCOL_DISTANCE_MATRIX}" >&2
      exit 1
      ;;
  esac
fi

append_pycol_parallel_args() {
  if [[ "${LIBRARY}" == "pycol" || "${LIBRARY}" == "both" ]]; then
    cmd+=(--pycol-distance-matrix "${PYCOL_DISTANCE_MATRIX}")
    if [[ "${PYCOL_PARALLEL_HEOM}" == "1" && "${PYCOL_DISTANCE_MATRIX}" != "skip" ]]; then
      cmd+=(--pycol-parallel-heom)
    fi
    if [[ "${PYCOL_PARALLEL_METRICS}" == "1" ]]; then
      cmd+=(--pycol-parallel-metrics)
    fi
  fi
}

append_complexity_max_rows_arg() {
  local per_dataset_max="${1:-}"
  if [[ -n "${per_dataset_max}" && "${per_dataset_max}" != "0" ]]; then
    cmd+=(--complexity-max-rows "${per_dataset_max}")
  elif [[ -n "${COMPLEXITY_MAX_ROWS:-}" && "${COMPLEXITY_MAX_ROWS}" != "0" ]]; then
    cmd+=(--complexity-max-rows "${COMPLEXITY_MAX_ROWS}")
  fi
}

print_cmd() {
  local src="$1" ref="$2" lbl="$3" per_max="${4:-}"
  local -a cmd=("${PYTHON}" "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --library "${LIBRARY}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
  )
  append_complexity_max_rows_arg "${per_max}"
  if [[ "${LIBRARY}" == "both" ]]; then
    cmd+=(--pycol-metrics "${PYCOL_METRICS_ARG}" --pymfe-metrics "${PYMFE_METRICS}")
    if [[ "${PYCOL_METRICS_ARG}" == "custom" ]]; then
      cmd+=(--pycol-custom-metrics "${PYCOL_CUSTOM_METRICS}")
    fi
  elif [[ "${LIBRARY}" == "pycol" ]]; then
    cmd+=(--metrics "${PYCOL_METRICS_ARG}")
    if [[ "${PYCOL_METRICS_ARG}" == "custom" ]]; then
      cmd+=(--pycol-custom-metrics "${PYCOL_CUSTOM_METRICS}")
    fi
  else
    cmd+=(--metrics "${METRICS_SINGLE}")
  fi
  append_pycol_parallel_args
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    cmd+=(--no-progress)
  fi
  printf '%q ' "${cmd[@]}"
  echo
}

run_one() {
  local src="$1" ref="$2" lbl="$3" per_max="${4:-}"
  local -a cmd=("${PYTHON}" "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --library "${LIBRARY}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
  )
  append_complexity_max_rows_arg "${per_max}"
  if [[ "${LIBRARY}" == "both" ]]; then
    cmd+=(--pycol-metrics "${PYCOL_METRICS_ARG}" --pymfe-metrics "${PYMFE_METRICS}")
    if [[ "${PYCOL_METRICS_ARG}" == "custom" ]]; then
      cmd+=(--pycol-custom-metrics "${PYCOL_CUSTOM_METRICS}")
    fi
  elif [[ "${LIBRARY}" == "pycol" ]]; then
    cmd+=(--metrics "${PYCOL_METRICS_ARG}")
    if [[ "${PYCOL_METRICS_ARG}" == "custom" ]]; then
      cmd+=(--pycol-custom-metrics "${PYCOL_CUSTOM_METRICS}")
    fi
  else
    cmd+=(--metrics "${METRICS_SINGLE}")
  fi
  append_pycol_parallel_args
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    cmd+=(--no-progress)
  fi
  "${cmd[@]}"
}

total="${#DATASETS[@]}"
echo "Datasets: ${total}  Output: ${OUTPUT_CSV}" >&2
if [[ "${LIBRARY}" == "pycol" ]] || [[ "${LIBRARY}" == "both" ]]; then
  echo "PyCol metrics: ${PYCOL_METRICS_ARG}  distance matrix: ${PYCOL_DISTANCE_MATRIX}  parallel HEOM: ${PYCOL_PARALLEL_HEOM}  parallel metrics: ${PYCOL_PARALLEL_METRICS}  n_jobs: ${N_JOBS}  max_rows(default): ${COMPLEXITY_MAX_ROWS}" >&2
fi

i=0
for entry in "${DATASETS[@]}"; do
  # Skip blank lines and comments
  [[ -z "${entry//[[:space:]]/}" ]] && continue
  [[ "${entry}" =~ ^[[:space:]]*# ]] && continue

  i=$((i + 1))
  IFS='|' read -r src ref lbl per_max <<<"${entry}"
  if [[ -z "${src:-}" || -z "${ref:-}" || -z "${lbl:-}" ]]; then
    echo "Skip malformed entry [${i}]: ${entry}" >&2
    continue
  fi

  echo "" >&2
  echo "========== [${i}/${total}] ${src}  ${ref} ==========" >&2
  if [[ -n "${per_max:-}" && "${per_max}" != "0" ]]; then
    echo "  complexity max rows (this dataset): ${per_max}" >&2
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${src}" "${ref}" "${lbl}" "${per_max:-}"
    continue
  fi

  set +e
  run_one "${src}" "${ref}" "${lbl}" "${per_max:-}"
  code=$?
  set -e

  if [[ "${code}" -ne 0 ]]; then
    echo "FAILED (exit ${code}): ${ref}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${code}"
    fi
  fi
done

echo "" >&2
echo "Batch finished. Output: ${OUTPUT_CSV}" >&2
