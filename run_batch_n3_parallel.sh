#!/usr/bin/env bash
#
# Run n3_parallel_cli.py once per dataset (PyCol N3 only, parallel sample loop).
# Edit the variables below, then:  chmod +x run_batch_n3_parallel.sh && ./run_batch_n3_parallel.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/n3_parallel_cli.py"

# Python: use project .venv if present, else python3 on PATH
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Shared CLI arguments (edit here)
# ---------------------------------------------------------------------------
# Parallel row chunks for N3 after dist_matrix is built (fork on Linux; use 1 on Windows).
N_JOBS="16"
MISSING_VALUES="impute_median"   # drop_rows | fill_zero | impute_median | impute_mean
OUTPUT_CSV="${SCRIPT_DIR}/results/batch_n3_parallel.csv"

# If > 0, subsample at most this many rows before Complexity (approximate N3; faster on large n).
COMPLEXITY_MAX_ROWS="0"

# k nearest neighbours for N3 (default 1 = Ho & Basu definition).
N3_K="1"

# IMB="1" → per-class N3; INST_LEVEL="1" → per-instance vector (disables parallel N3 in fork).
IMB="0"
INST_LEVEL="0"

# CONTINUE_ON_ERROR: "0" = stop on first failure; "1" = continue
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# DRY_RUN: "1" = print commands only (override: DRY_RUN=1 ./run_batch_n3_parallel.sh)
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Dataset list (same format as run_batch_parallel.sh)
# Format:   source|ref|label_column
# ---------------------------------------------------------------------------
DATASETS=(
  "uci|https://archive.ics.uci.edu/dataset/186/wine+quality|target"
  "uci|https://archive.ics.uci.edu/dataset/2/adult|target"
  "uci|https://archive.ics.uci.edu/dataset/222/bank+marketing|target"
  "uci|https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping|target"
  "uci|https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators|target"
  "uci|https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset|target"
  "uci|https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume|target"
  # "csv|/absolute/path/to/data.csv|class"
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "${OUTPUT_CSV}")"

print_cmd() {
  local src="$1" ref="$2" lbl="$3"
  local -a cmd=("${PYTHON}" "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
    --k "${N3_K}"
  )
  if [[ -n "${COMPLEXITY_MAX_ROWS:-}" && "${COMPLEXITY_MAX_ROWS}" != "0" ]]; then
    cmd+=(--complexity-max-rows "${COMPLEXITY_MAX_ROWS}")
  fi
  if [[ "${IMB}" == "1" ]]; then
    cmd+=(--imb)
  fi
  if [[ "${INST_LEVEL}" == "1" ]]; then
    cmd+=(--inst-level)
  fi
  printf '%q ' "${cmd[@]}"
  echo
}

run_one() {
  local src="$1" ref="$2" lbl="$3"
  local -a cmd=("${PYTHON}" "${CLI}"
    --source "${src}"
    --ref "${ref}"
    --label-column "${lbl}"
    --n-jobs "${N_JOBS}"
    --missing-values "${MISSING_VALUES}"
    --output-csv "${OUTPUT_CSV}"
    --k "${N3_K}"
  )
  if [[ -n "${COMPLEXITY_MAX_ROWS:-}" && "${COMPLEXITY_MAX_ROWS}" != "0" ]]; then
    cmd+=(--complexity-max-rows "${COMPLEXITY_MAX_ROWS}")
  fi
  if [[ "${IMB}" == "1" ]]; then
    cmd+=(--imb)
  fi
  if [[ "${INST_LEVEL}" == "1" ]]; then
    cmd+=(--inst-level)
  fi
  "${cmd[@]}"
}

total="${#DATASETS[@]}"
echo "N3 batch: ${total} dataset(s)  Python: ${PYTHON}" >&2
echo "Output: ${OUTPUT_CSV}  n_jobs=${N_JOBS}  k=${N3_K}" >&2

i=0
for entry in "${DATASETS[@]}"; do
  [[ -z "${entry//[[:space:]]/}" ]] && continue
  [[ "${entry}" =~ ^[[:space:]]*# ]] && continue

  i=$((i + 1))
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
