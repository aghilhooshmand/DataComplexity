#!/usr/bin/env bash
#
# Run parallel_complexity_cli.py once per dataset, in order.
# Edit the variables below (defaults + dataset list), then:  chmod +x run_batch_parallel.sh && ./run_batch_parallel.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/parallel_complexity_cli.py"

# ---------------------------------------------------------------------------
# Shared CLI arguments (edit here)
# ---------------------------------------------------------------------------
LIBRARY="both"                    # pycol | pymfe | both
N_JOBS="16"
MISSING_VALUES="impute_median"   # drop_rows | fill_zero | impute_median | impute_mean
OUTPUT_CSV="${SCRIPT_DIR}/results/batch_parallel_complexity.csv"

# When LIBRARY is "both":
PYCOL_METRICS="all"
PYMFE_METRICS="all"

# When LIBRARY is "pycol" or "pymfe" only, set METRICS_SINGLE instead (and adjust build_cmd):
METRICS_SINGLE="all"

# Pass --no-progress to each run? 1=yes, 0=no
NO_PROGRESS="0"

# If a dataset fails, continue with the next? 1=yes, 0=stop
CONTINUE_ON_ERROR="0"

# Dry run: print commands only (1=yes)
DRY_RUN="0"

# ---------------------------------------------------------------------------
# Dataset list: one entry per line inside the array.
# Format:   source|ref|label_column
#   source = uci | openml | csv
#   ref     = UCI/OpenML URL or id, OR absolute path to CSV when source=csv
#   label_column = target column name (for uci/openml from this CLI, use "target")
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

print_cmd() {
  local src="$1" ref="$2" lbl="$3"
  local -a cmd=(python3 "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --library "${LIBRARY}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
  )
  if [[ "${LIBRARY}" == "both" ]]; then
    cmd+=(--pycol-metrics "${PYCOL_METRICS}" --pymfe-metrics "${PYMFE_METRICS}")
  else
    cmd+=(--metrics "${METRICS_SINGLE}")
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    cmd+=(--no-progress)
  fi
  printf '%q ' "${cmd[@]}"
  echo
}

run_one() {
  local src="$1" ref="$2" lbl="$3"
  local -a cmd=(python3 "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --library "${LIBRARY}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
  )
  if [[ "${LIBRARY}" == "both" ]]; then
    cmd+=(--pycol-metrics "${PYCOL_METRICS}" --pymfe-metrics "${PYMFE_METRICS}")
  else
    cmd+=(--metrics "${METRICS_SINGLE}")
  fi
  if [[ "${NO_PROGRESS}" == "1" ]]; then
    cmd+=(--no-progress)
  fi
  "${cmd[@]}"
}

total="${#DATASETS[@]}"
echo "Datasets: ${total}  Output: ${OUTPUT_CSV}" >&2

i=0
for entry in "${DATASETS[@]}"; do
  # Skip blank lines and comments
  [[ -z "${entry//[[:space:]]/}" ]] && continue
  [[ "${entry}" =~ ^[[:space:]]*# ]] && continue

  ((++i))
  IFS='|' read -r src ref lbl <<<"${entry}"
  if [[ -z "${src:-}" || -z "${ref:-}" || -z "${lbl:-}" ]]; then
    echo "Skip malformed entry [${i}]: ${entry}" >&2
    continue
  fi

  echo "" >&2
  echo "========== [${i}/${total}] ${src}  ${ref} ==========" >&2

  if [[ "${DRY_RUN}" == "1" ]]; then
    print_cmd "${src}" "${ref}" "${lbl}"
    continue
  fi

  set +e
  run_one "${src}" "${ref}" "${lbl}"
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
