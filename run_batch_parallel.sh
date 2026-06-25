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
# Tuned for Hive: full sample, ALL PyCol metrics, HEOM matrices in RAM (float64).
# ---------------------------------------------------------------------------
LIBRARY="pycol"                    # pycol | pymfe | both
# Worker cap for HEOM rows (--pycol-parallel-heom). Keep moderate (24–32) when preset=all
# to limit peak RAM during parallel matrix build; CLI caps to min(N_JOBS, metrics, CPUs).
N_JOBS="32"
MISSING_VALUES="impute_median"   # drop_rows | fill_zero | impute_median | impute_mean
# Append/upsert into results summary (dataset_file column).
OUTPUT_CSV="${SCRIPT_DIR}/results/datasets_complexity_summary.csv"
UPSERT_KEY="dataset_file"        # dataset_name | dataset_file (must match OUTPUT_CSV columns)

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
# Active setting: "all" → full catalog incl. T1, NSG, ICSV (two HEOM matrices).
PYCOL_METRICS_ARG="all"
# Ignored while PYCOL_METRICS_ARG is not "custom" (example list for when you switch to custom):
PYCOL_CUSTOM_METRICS="F1,F2,F3,F4,F1v,input_noise,purity,N2,N3,C1,C2"

# HEOM matrices: always RAM float64 (stock PyCol precision).
PYCOL_MATRIX_DTYPE="float64"

# PyCol HEOM tier (only when LIBRARY is pycol or both):
#   auto  — from preset: all → both matrices (dist + unnorm for T1/NSG/ICSV)
#   skip | dist | both — force tier
PYCOL_DISTANCE_MATRIX="auto"

# Parallel HEOM row workers. "1" = on; "0" = serial.
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
#   "1" = log the error and continue with the next dataset (default for Hive/server)
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Append per-dataset failures here (exit code + stderr/stdout).
FAILURE_LOG="${SCRIPT_DIR}/results/batch_failures.log"
# Full batch timeline (START/OK/FAIL/INTERRUPT for every dataset).
BATCH_RUN_LOG="${SCRIPT_DIR}/results/batch_run.log"
# Per-dataset stdout+stderr (one file per CSV name).
BATCH_LOG_DIR="${SCRIPT_DIR}/results/batch_logs"

# DRY_RUN:
#   "0" = really run parallel_complexity_cli.py for each line
#   "1" = only print the shell-quoted command that would be run (no execution)
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Dataset list: one entry per line inside the array.
# Format:   source|ref|label_column|[max_rows]
#   source = uci | openml | csv
#   ref     = UCI/OpenML URL or id, OR path to CSV when source=csv (${SCRIPT_DIR}/pmlb_DS/…)
#   label_column = target column name (PMLB encoded CSVs use "target")
#   max_rows (optional) = override COMPLEXITY_MAX_ROWS for this line only
# Use | as separator (do not use | inside URLs).
#
# Hive batch: 56/63 done. 7 remain (largest; batch stopped at ring.csv).
# coil2000 done earlier (old run order). Upsert skips rows with pycol_F1.
# ---------------------------------------------------------------------------
DATASETS=(
  "csv|${SCRIPT_DIR}/pmlb_DS/ring.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/twonorm.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/mushroom.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/pendigits.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/nursery.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/magic.csv|target"
  "csv|${SCRIPT_DIR}/pmlb_DS/letter.csv|target"
  # Legacy UCI examples:
  # "uci|https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic|target"
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
    cmd+=(--pycol-matrix-dtype "${PYCOL_MATRIX_DTYPE}")
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
    --upsert-key "${UPSERT_KEY}"
    --failure-log "${FAILURE_LOG}"
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
    --upsert-key "${UPSERT_KEY}"
    --failure-log "${FAILURE_LOG}"
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
ok_count=0
fail_count=0
CURRENT_DATASET=""
BATCH_SESSION_ID="$(date -Iseconds)"

mkdir -p "$(dirname "${OUTPUT_CSV}")" "${BATCH_LOG_DIR}"

log_batch() {
  local line
  line="$(date -Iseconds) ${*}"
  echo "${line}" >>"${BATCH_RUN_LOG}"
  echo "${line}" >&2
}

append_failure_block() {
  {
    echo "========== $(date -Iseconds) ${*} =========="
    echo "session: ${BATCH_SESSION_ID}"
    echo "dataset: ${CURRENT_DATASET}"
    echo "ref: ${CURRENT_REF:-}"
    echo "command: ${LAST_CMD:-}"
    echo ""
  } >>"${FAILURE_LOG}"
}

on_batch_signal() {
  local sig="$1"
  log_batch "SIGNAL ${sig} during dataset=${CURRENT_DATASET:-unknown}"
  append_failure_block "INTERRUPTED signal=${sig} exit=130"
  exit 130
}

trap 'on_batch_signal INT' INT
trap 'on_batch_signal TERM' TERM

{
  echo ""
  echo "################################################################"
  echo "# BATCH SESSION ${BATCH_SESSION_ID}"
  echo "# host: $(hostname)  pid: $$"
  echo "# datasets: ${total}  output: ${OUTPUT_CSV}"
  echo "################################################################"
} >>"${BATCH_RUN_LOG}"

echo "Datasets: ${total}  Output: ${OUTPUT_CSV}" >&2
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}" >&2
echo "Logs: failures=${FAILURE_LOG}  run=${BATCH_RUN_LOG}  per-dataset=${BATCH_LOG_DIR}/" >&2
if [[ "${LIBRARY}" == "pycol" ]] || [[ "${LIBRARY}" == "both" ]]; then
  echo "PyCol metrics: ${PYCOL_METRICS_ARG}  matrix dtype: ${PYCOL_MATRIX_DTYPE}  distance matrix: ${PYCOL_DISTANCE_MATRIX}  parallel HEOM: ${PYCOL_PARALLEL_HEOM}  parallel metrics: ${PYCOL_PARALLEL_METRICS}  n_jobs: ${N_JOBS}  max_rows(default): ${COMPLEXITY_MAX_ROWS}" >&2
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

  CURRENT_DATASET="$(basename "${ref}")"
  CURRENT_REF="${ref}"
  ds_log="${BATCH_LOG_DIR}/${CURRENT_DATASET}.log"
  started_at="$(date -Iseconds)"
  started_epoch="${SECONDS}"

  log_batch "START [${i}/${total}] ${CURRENT_DATASET} ref=${ref}"
  {
    echo "========== START ${started_at} [${i}/${total}] =========="
    echo "ref: ${ref}"
    print_cmd "${src}" "${ref}" "${lbl}" "${per_max:-}"
    echo ""
  } >>"${ds_log}"

  LAST_CMD="$(print_cmd "${src}" "${ref}" "${lbl}" "${per_max:-}")"
  set +e
  run_one "${src}" "${ref}" "${lbl}" "${per_max:-}" 2>&1 | tee -a "${ds_log}" >&2
  code=${PIPESTATUS[0]}
  set -e
  elapsed="$((SECONDS - started_epoch))"

  {
    echo ""
    echo "========== END $(date -Iseconds) exit=${code} elapsed=${elapsed}s =========="
  } >>"${ds_log}"

  if [[ "${code}" -eq 0 ]]; then
    ok_count=$((ok_count + 1))
    log_batch "OK    [${i}/${total}] ${CURRENT_DATASET} elapsed=${elapsed}s"
  else
    fail_count=$((fail_count + 1))
    echo "FAILED (exit ${code}): ${ref}" >&2
    oom_hint=""
    if [[ "${code}" -eq 137 ]] || [[ "${code}" -eq 9 ]]; then
      oom_hint="Likely OOM kill (SIGKILL) — distance matrix may exceed RAM."
    elif [[ "${code}" -eq 134 ]]; then
      oom_hint="Likely abort (SIGABRT) — possible memory or native crash."
    fi
    append_failure_block "FAILED exit=${code} elapsed=${elapsed}s ${oom_hint}"
    tail -n 80 "${ds_log}" >>"${FAILURE_LOG}"
    echo "" >>"${FAILURE_LOG}"
    if [[ -n "${oom_hint}" ]]; then
      echo "${oom_hint}" >>"${FAILURE_LOG}"
      echo "" >>"${FAILURE_LOG}"
    fi
    log_batch "FAIL  [${i}/${total}] ${CURRENT_DATASET} exit=${code} elapsed=${elapsed}s ${oom_hint}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${code}"
    fi
  fi
  CURRENT_DATASET=""
  CURRENT_REF=""
done

echo "" >&2
echo "Batch finished. Output: ${OUTPUT_CSV}" >&2
echo "Succeeded: ${ok_count}  Failed: ${fail_count}  Total: ${total}" >&2
log_batch "FINISH session=${BATCH_SESSION_ID} ok=${ok_count} fail=${fail_count} total=${total}"
if [[ "${fail_count}" -gt 0 ]]; then
  echo "See failure details: ${FAILURE_LOG}" >&2
  echo "Per-dataset logs: ${BATCH_LOG_DIR}/" >&2
fi
