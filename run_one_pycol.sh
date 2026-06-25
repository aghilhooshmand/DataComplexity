#!/usr/bin/env bash
#
# Run PyCol (preset=all, full sample) for ONE dataset — serial, no parallel HEOM/metrics.
#
# Usage:
#   ./run_one_pycol.sh ring
#   ./run_one_pycol.sh mushroom.csv
#   ./run_one_pycol.sh --list          # show names under pmlb_DS/
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
FAILURE_LOG="${FAILURE_LOG:-${SCRIPT_DIR}/results/batch_failures.log}"
BATCH_LOG_DIR="${BATCH_LOG_DIR:-${SCRIPT_DIR}/results/batch_logs}"
PYCOL_MEMMAP_DIR="${PYCOL_MEMMAP_DIR:-${SCRIPT_DIR}/results/pycol_memmap}"
MISSING_VALUES="${MISSING_VALUES:-impute_median}"
LABEL_COLUMN="${LABEL_COLUMN:-target}"
METRICS="${METRICS:-all}"
COMPLEXITY_MAX_ROWS="${COMPLEXITY_MAX_ROWS:-0}"
PYCOL_DISTANCE_MATRIX="${PYCOL_DISTANCE_MATRIX:-auto}"
PYCOL_MATRIX_STORAGE="${PYCOL_MATRIX_STORAGE:-auto}"
PYCOL_MATRIX_DTYPE="${PYCOL_MATRIX_DTYPE:-float64}"
PYCOL_MEMMAP_THRESHOLD_N="${PYCOL_MEMMAP_THRESHOLD_N:-8145}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <dataset>

  <dataset>  basename without path, e.g. ring or ring.csv
             (file must exist under ${PMLB_DIR}/)

Serial PyCol: n_jobs=1, no parallel HEOM, no parallel metrics.
Results append to: ${OUTPUT_CSV}

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

mkdir -p "$(dirname "${OUTPUT_CSV}")" "${PYCOL_MEMMAP_DIR}" "${BATCH_LOG_DIR}"

cmd=(
  "${PYTHON}" "${CLI}"
  --source csv
  --ref "${CSV_PATH}"
  --label-column "${LABEL_COLUMN}"
  --library pycol
  --metrics "${METRICS}"
  --n-jobs 1
  --missing-values "${MISSING_VALUES}"
  --output-csv "${OUTPUT_CSV}"
  --upsert-key "${UPSERT_KEY}"
  --failure-log "${FAILURE_LOG}"
  --pycol-distance-matrix "${PYCOL_DISTANCE_MATRIX}"
  --pycol-matrix-storage "${PYCOL_MATRIX_STORAGE}"
  --pycol-matrix-dtype "${PYCOL_MATRIX_DTYPE}"
  --pycol-memmap-threshold-n "${PYCOL_MEMMAP_THRESHOLD_N}"
  --pycol-memmap-dir "${PYCOL_MEMMAP_DIR}"
)

if [[ -n "${COMPLEXITY_MAX_ROWS}" && "${COMPLEXITY_MAX_ROWS}" != "0" ]]; then
  cmd+=(--complexity-max-rows "${COMPLEXITY_MAX_ROWS}")
fi

echo "Dataset: ${DATASET_FILE}" >&2
echo "Output:  ${OUTPUT_CSV}" >&2
echo "Mode:    serial PyCol (n_jobs=1, no parallel HEOM/metrics)" >&2

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

ds_log="${BATCH_LOG_DIR}/${DATASET_FILE}.log"
started_at="$(date -Iseconds)"
started_epoch="${SECONDS}"

{
  echo "========== START ${started_at} =========="
  printf '%q ' "${cmd[@]}"
  echo
  echo ""
} >>"${ds_log}"

set +e
"${cmd[@]}" 2>&1 | tee -a "${ds_log}" >&2
code=${PIPESTATUS[0]}
set -e
elapsed="$((SECONDS - started_epoch))"

{
  echo ""
  echo "========== END $(date -Iseconds) exit=${code} elapsed=${elapsed}s =========="
} >>"${ds_log}"

if [[ "${code}" -ne 0 ]]; then
  echo "FAILED (exit ${code}): ${DATASET_FILE}" >&2
  exit "${code}"
fi

echo "OK: ${DATASET_FILE} (${elapsed}s)" >&2
