#!/usr/bin/env bash
#
# Run PyCol for ONE dataset at a time (Hive-friendly).
# Default preset: cheap — skips T1, NSG, ICSV (multi-day on large n).
# Parallelism is automatic: cores from system load, shared HEOM matrix, parallel metrics.
#
# Usage:
#   ./run_one_pycol.sh ring
#   ./run_one_pycol.sh mushroom.csv
#   ./run_one_pycol.sh --list
#   METRICS=all ./run_one_pycol.sh ring    # full catalog (29 metrics; very slow on large n)
#   METRICS=middle ./run_one_pycol.sh ring # cheap + ONB + DBC
#   DRY_RUN=1 ./run_one_pycol.sh magic
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/parallel_complexity_cli.py"
PMLB_DIR="${SCRIPT_DIR}/pmlb_DS"

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

# Defaults (override via env if needed)
OUTPUT_CSV="${OUTPUT_CSV:-${SCRIPT_DIR}/results/datasets_complexity_summary.csv}"
UPSERT_KEY="${UPSERT_KEY:-dataset_file}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/results/logs}"
LOCK_DIR="${LOCK_DIR:-${SCRIPT_DIR}/results/locks}"
MISSING_VALUES="${MISSING_VALUES:-impute_median}"
LABEL_COLUMN="${LABEL_COLUMN:-target}"
# Default preset: cheap — 24 metrics (skips T1, NSG, ICSV, ONB, DBC).
# Use METRICS=middle for ONB/DBC; METRICS=full or METRICS=all for all 29.
METRICS="${METRICS:-cheap}"
COMPLEXITY_MAX_ROWS="${COMPLEXITY_MAX_ROWS:-0}"
PYCOL_DISTANCE_MATRIX="${PYCOL_DISTANCE_MATRIX:-auto}"
PYCOL_MATRIX_DTYPE="${PYCOL_MATRIX_DTYPE:-float64}"
# 0 = auto: CLI picks cores from cpu_count minus current load (Linux getloadavg).
N_JOBS="${N_JOBS:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <dataset>

  <dataset>  basename without path, e.g. ring or ring.csv
             (file must exist under ${PMLB_DIR}/)

One dataset at a time. Default METRICS=cheap (24 metrics; skips T1, NSG, ICSV, ONB, DBC).
Use METRICS=middle for ONB/DBC; METRICS=full for all 29.
Parallel HEOM + shared-matrix metrics are automatic.
N_JOBS=0 uses available cores minus system load.
Results: ${OUTPUT_CSV}
Log:     ${LOG_DIR}/<dataset>.csv.log

Options:
  --list     print available *.csv names in pmlb_DS/
  -h, --help this message
EOF
}

list_datasets() {
  shopt -s nullglob
  local f base
  for f in "${PMLB_DIR}"/*.csv; do
    base="$(basename "${f}")"
    echo "${base%.csv}"
  done | sort
}

resolve_csv() {
  local name="$1"
  name="${name%.csv}"
  local path="${PMLB_DIR}/${name}.csv"
  if [[ ! -f "${path}" ]]; then
    echo "Dataset not found: ${path}" >&2
    echo "Try: $(basename "$0") --list" >&2
    exit 1
  fi
  printf '%s\n' "${path}"
}

acquire_lock() {
  local lock_file="$1"
  mkdir -p "$(dirname "${lock_file}")"
  if [[ -f "${lock_file}" ]]; then
    local old_pid
    old_pid="$(cat "${lock_file}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "ERROR: ${DATASET_FILE} is already running (pid ${old_pid})." >&2
      echo "       Stop it first: kill ${old_pid}" >&2
      echo "       Lock: ${lock_file}" >&2
      exit 1
    fi
    echo "Removing stale lock for ${DATASET_FILE} (pid ${old_pid:-?} not running)." >&2
    rm -f "${lock_file}"
  fi
  echo "$$" >"${lock_file}"
}

release_lock() {
  rm -f "$1"
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
  --list)
    list_datasets
    exit 0
    ;;
esac

DATASET_ARG="$1"
CSV_PATH="$(resolve_csv "${DATASET_ARG}")"
DATASET_FILE="$(basename "${CSV_PATH}")"
DS_LOG="${LOG_DIR}/${DATASET_FILE}.log"
LOCK_FILE="${LOCK_DIR}/${DATASET_FILE}.lock"

mkdir -p "$(dirname "${OUTPUT_CSV}")" "${LOG_DIR}"

cmd=(
  "${PYTHON}" "${CLI}"
  --source csv
  --ref "${CSV_PATH}"
  --label-column "${LABEL_COLUMN}"
  --library pycol
  --metrics "${METRICS}"
  --n-jobs "${N_JOBS}"
  --missing-values "${MISSING_VALUES}"
  --output-csv "${OUTPUT_CSV}"
  --upsert-key "${UPSERT_KEY}"
  --failure-log "${DS_LOG}"
  --pycol-distance-matrix "${PYCOL_DISTANCE_MATRIX}"
  --pycol-matrix-dtype "${PYCOL_MATRIX_DTYPE}"
)

if [[ -n "${COMPLEXITY_MAX_ROWS}" && "${COMPLEXITY_MAX_ROWS}" != "0" ]]; then
  cmd+=(--complexity-max-rows "${COMPLEXITY_MAX_ROWS}")
fi

echo "Dataset: ${DATASET_FILE}" >&2
echo "Output:  ${OUTPUT_CSV}" >&2
echo "Log:     ${DS_LOG}" >&2
echo "Metrics: ${METRICS} (cheap=24, middle=+ONB/DBC, full=all 29)" >&2
if [[ "${N_JOBS}" == "0" ]]; then
  echo "Mode:    auto n_jobs (cpus − load); parallel HEOM + shared metrics" >&2
else
  echo "Mode:    n_jobs=${N_JOBS}; parallel HEOM + shared metrics" >&2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

acquire_lock "${LOCK_FILE}"
trap 'release_lock "${LOCK_FILE}"' EXIT

started_at="$(date -Iseconds)"
started_epoch="${SECONDS}"

{
  echo "========== START ${started_at} =========="
  printf '%q ' "${cmd[@]}"
  echo
  echo ""
} >>"${DS_LOG}"

set +e
"${cmd[@]}" 2>&1 | tee -a "${DS_LOG}" >&2
code=${PIPESTATUS[0]}
set -e
elapsed="$((SECONDS - started_epoch))"

{
  echo ""
  echo "========== END $(date -Iseconds) exit=${code} elapsed=${elapsed}s =========="
} >>"${DS_LOG}"

if [[ "${code}" -ne 0 ]]; then
  echo "FAILED (exit ${code}): ${DATASET_FILE} — see ${DS_LOG}" >&2
  exit "${code}"
fi

echo "OK: ${DATASET_FILE} (${elapsed}s) — log: ${DS_LOG}" >&2
