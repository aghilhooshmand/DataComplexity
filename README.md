# Dataset Complexity

Analyze classification datasets with **PyCol** (`pycol-complexity`) and **PyMFE** (complexity group). Use the **Streamlit app** for interactive runs, or the **CLI / batch script** for servers and many datasets.

---

## Install

**Requirements:** Python 3.10+ (tested on 3.13), `pip`.

```bash
cd /path/to/DataComplexity
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to use

### Streamlit app (one dataset or comparison)

```bash
source .venv/bin/activate
streamlit run app.py
```

| Page | What it does |
|------|----------------|
| **Complexity Calculator** | One dataset (CSV / UCI / OpenML), metrics, CSV download, t-SNE |
| **Dataset Comparison** | Several datasets, one comparison table and charts |
| **Metric Reference** | Short descriptions of supported metrics |

**Typical flow (Calculator):**

1. Choose data source and load the dataset.
2. Pick the **label** column and **missing-value** strategy (`impute_median` is a good default).
3. Choose **pycol** / **pymfe** and a **PyCol preset** (start with **`cheap_minimal`**).
4. For PyCol, leave **distance matrix → Skip** unless you need N2, N3, etc. (the app shows RAM/time warnings).
5. Click **Compute complexity** and download the CSV.

### Batch run (many UCI / OpenML / CSV datasets)

Edit the dataset list and options at the top of **`run_batch_parallel.sh`**, then:

```bash
chmod +x run_batch_parallel.sh
./run_batch_parallel.sh
```

Results are appended/updated in **`results/batch_parallel_complexity.csv`** (one row per dataset).

**Sensible defaults in the script:**

```bash
PYCOL_METRICS_ARG="cheap_minimal"
PYCOL_DISTANCE_MATRIX="skip"
```

Dry-run (print commands only):

```bash
DRY_RUN=1 ./run_batch_parallel.sh
```

---

## Advanced

### CLI (`parallel_complexity_cli.py`, version 1.7.0)

Command-line runner for **one dataset**. Writes one CSV row (upserts by `dataset_name` when you reuse the same output file).

```bash
python3 parallel_complexity_cli.py --version
```

**Main options:**

| Option | Purpose |
|--------|---------|
| `--source` | `csv`, `uci`, `openml` |
| `--ref` | File path, dataset id, or full UCI/OpenML URL |
| `--label-column` | Target column name |
| `--library` | `pycol`, `pymfe`, or `both` |
| `--metrics` / `--pycol-metrics` | Preset name, `all`, `custom`, or comma-separated list |
| `--pycol-custom-metrics` | With `custom`, e.g. `F1,N3,N1` |
| `--pycol-distance-matrix` | `skip` (default) or `build` |
| `--pycol-parallel-heom` | Row-parallel HEOM when building the matrix |
| `--complexity-max-rows` | Subsample rows for metrics only (`0` = all rows) |
| `--missing-values` | `impute_median` (default), `impute_mean`, `fill_zero`, `drop_rows` |
| `--n-jobs` | CPU cap for parallel metric workers / parallel HEOM |
| `--no-progress` | Quiet mode for logs and batch |

**Example — fast PyCol (no distance matrix):**

```bash
python3 parallel_complexity_cli.py \
  --source uci \
  --ref "https://archive.ics.uci.edu/dataset/186/wine+quality" \
  --label-column target \
  --library pycol \
  --metrics cheap_minimal \
  --pycol-distance-matrix skip \
  --missing-values impute_median \
  --output-csv results/wine.csv
```

**Example — neighbor metrics with subsampling:**

```bash
python3 parallel_complexity_cli.py \
  --source uci \
  --ref 2 \
  --library pycol \
  --metrics cheap \
  --pycol-distance-matrix build \
  --pycol-parallel-heom \
  --complexity-max-rows 8000 \
  --n-jobs 8 \
  --output-csv results/adult_cheap.csv
```

On large **n**, PyCol metrics run **sequentially in one process** by default (≥5000 rows after cleaning) to avoid duplicating huge distance structures. Use **`--pycol-parallel-metrics`** only on smaller datasets if you want one process per metric.

---

### PyCol metric presets

Same names in Streamlit, CLI, and **`run_batch_parallel.sh`**:

| Preset | Metrics (summary) | Distance matrix |
|--------|-------------------|-----------------|
| **`cheap_minimal`** | F1, F2, F3, F4, F1v, input_noise, purity | Not required |
| **`cheap`** | minimal + N2, N3, C1, C2 | Required |
| **`expensive_core`** | N1, N4, T1, LSC, kDN, borderline | Required |
| **`expensive`** | All PyCol metrics not in `cheap` | Mixed |
| **`all`** | Full PyCol catalog | Most need matrix |
| **`custom`** | Your comma-separated list | Depends on selection |

---

### Distance matrix (HEOM)

PyCol’s default path builds two **n×n** matrices (HEOM) in `Complexity.__init__` — **O(n²)** time and RAM.

| Mode | Flag / setting | Behavior |
|------|----------------|----------|
| **Skip** | `--pycol-distance-matrix skip` / `PYCOL_DISTANCE_MATRIX=skip` | No matrix. Only metrics that use X/y directly. Others are **omitted** (see `pycol_metrics_omitted_need_distance` in output). |
| **Build** | `build` | Full matrix; required for N2, N3, kDN, N1, etc. |

**RAM estimate (both matrices, float64):**

```text
RAM ≈ 16 × n² bytes   (e.g. n=10,000 → ~1.6 GB; n=50,000 → ~40 GB)
```

For large tables, use **`cheap_minimal` + skip**, or **`--complexity-max-rows`** / **`COMPLEXITY_MAX_ROWS`** in the batch script.

---

### Parallel HEOM (`pycol_heom.py`)

Optional replacement for PyCol’s sequential `__distance_HEOM` (same definition, validated on Iris).

- **CLI:** `--pycol-parallel-heom` (only with `build`)
- **Batch:** `PYCOL_PARALLEL_HEOM="1"` when `PYCOL_DISTANCE_MATRIX=build`
- **Streamlit:** checkbox under **Build** distance matrix

Uses row chunks and **`--n-jobs`** workers. **Does not reduce RAM** — still stores full n×n matrices.

Implementation: [`pycol_heom.py`](pycol_heom.py), wired in [`complexity_core.py`](complexity_core.py) via `build_pycol_complexity()`.

---

### Parallel metrics vs one Complexity instance

| Mechanism | What runs in parallel | RAM note |
|-----------|----------------------|----------|
| **`--pycol-parallel-heom`** | Building the distance matrix once | One matrix (if `build`) |
| **`--pycol-parallel-metrics`** | Each metric in its own process | Each worker may build its **own** matrix — high RAM on large **n** |
| **Default (large n)** | Metrics sequential, **one** `Complexity` | Lowest peak RAM |

Prefer **one build + sequential metrics** (or **skip** matrix) on big datasets.

---

### Batch script variables (`run_batch_parallel.sh`)

| Variable | Meaning |
|----------|---------|
| `LIBRARY` | `pycol`, `pymfe`, or `both` |
| `PYCOL_METRICS_ARG` | Preset or comma list |
| `PYCOL_DISTANCE_MATRIX` | `skip` or `build` |
| `PYCOL_PARALLEL_HEOM` | `1` adds `--pycol-parallel-heom` when building |
| `COMPLEXITY_MAX_ROWS` | `0` = all rows; else random subsample for metrics |
| `N_JOBS` | Parallel workers cap |
| `MISSING_VALUES` | Same as CLI |
| `DATASETS` | Lines: `uci\|URL\|target` or `csv\|/path/file.csv\|label` |
| `DRY_RUN` | `1` = print commands only |
| `CONTINUE_ON_ERROR` | `1` = continue after a failed dataset |

Uses **`.venv/bin/python`** automatically when present.

---

### Missing values (preprocessing)

1. Clean tokens (`?`, empty, etc.) → NaN  
2. One-hot encode categoricals  
3. Coerce features to numeric  

Then apply your strategy on **feature** NaNs only (labels with missing target are always dropped):

| Strategy | Effect |
|----------|--------|
| **`impute_median`** (default) | Column medians, then 0 for empty columns |
| **`impute_mean`** | Column means |
| **`fill_zero`** | NaN → 0 |
| **`drop_rows`** | Drop any row with a feature NaN |

**UCI Adult:** prefer **`impute_median`** — `drop_rows` can remove almost all rows.

---

### Metric columns in CSV

- `pycol_*` — PyCol metrics  
- `pymfe_*` — PyMFE metrics  
- `pycol_distance_matrix_skipped`, `pycol_metrics_omitted_need_distance`, `pycol_metrics_preset` — run metadata when applicable  

---

### Troubleshooting

- **Import errors:** activate `.venv` and `pip install -r requirements.txt`
- **UCI/OpenML load fails:** check id/URL and network
- **Out of memory / very slow PyCol:** use `cheap_minimal` + `skip`, or lower `COMPLEXITY_MAX_ROWS`
- **Metrics missing in output with `skip`:** they need the distance matrix — switch to `build` or change preset
- **Streamlit UI stale:** refresh the browser tab

---

## Maintainer

- **Aghil Hooshmand** — Research Fellow, FORGE project  
- **Group:** Biocomputing and Development Systems (BDS), University of Limerick  
- **Contact:** `aghil.hooshmand@ul.ie`
