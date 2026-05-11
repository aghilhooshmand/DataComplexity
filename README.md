# Dataset Complexity App

A Streamlit application for dataset complexity analysis using:

- `pycol-complexity`
- `pymfe` (complexity group)

The app supports:

- single-dataset analysis
- multi-dataset comparison
- dataset loading from CSV upload, UCI, and OpenML
- **configurable missing-value handling** for features (same options in Streamlit and CLI)
- t-SNE visualization
- CSV export for results
- parallel CLI execution for one dataset (speedup on multi-core servers)
- shell batch script (`run_batch_parallel.sh`) to run the CLI over many datasets sequentially

---

## Missing values (what they are and where to set them)

Complexity metrics need a **numeric feature matrix** \(X\) and **integer class labels** \(y\). Raw tables often have missing or placeholder values (for example `?` in UCI Adult, empty cells, or invalid strings). This project applies a **fixed preprocessing pipeline**, then your chosen strategy fixes any remaining **NaN in feature columns only**.

### Preprocessing order (always the same)

1. **Token cleanup:** In string/object columns, values like `?`, empty string, `nan`, `None` (as text) are turned into true missing (NaN).
2. **Categorical encoding:** Object/category feature columns are expanded with **one-hot encoding** (`pandas.get_dummies`, including a bucket for missing categories where applicable).
3. **Numeric coercion:** Every feature column is converted with `to_numeric(..., errors="coerce")`, so anything that is not a number becomes NaN.

After step 3, some cells may still be NaN. **You choose** what happens next using one of the strategies below.

### The four strategies (feature columns only)

| Internal name | User-facing idea | What happens |
|---------------|------------------|----------------|
| **`impute_median`** | Univariate median imputation (default) | Each feature column: missing cells are filled with that column’s **median** over non-missing rows. If an entire column is still missing (no non-NaN values), it is filled with **0**. Good default for messy real-world CSV/UCI data. |
| **`impute_mean`** | Univariate mean imputation | Same as median, but using the column **mean** instead of median. |
| **`fill_zero`** | Fill with zero | Every remaining NaN in features is set to **0**. Simple and fast; assumes “missing means zero” which may or may not suit your domain. |
| **`drop_rows`** | Listwise deletion | Any row that still has **at least one** NaN in **any** feature column is **removed**. Strict and transparent, but on sparse or wide data you can lose many rows (or all rows if every row has some missing cell). |

### Labels (target column)

- Rows with a **missing label** are **always dropped**, regardless of strategy.
- The chosen strategy is recorded in outputs as the column **`missing_values`** where that row is built via `basic_info_row` (Streamlit) or set explicitly (CLI).

### Where to choose the strategy

| Interface | Where |
|-----------|--------|
| **Streamlit — Complexity Calculator** | After choosing the label column: **“Missing values in features (after encoding)”**. Used for the dataset summary, metric computation, and t-SNE. |
| **Streamlit — Dataset Comparison** | Control above libraries/metrics: same dropdown; applies to **every** dataset in the comparison list for metrics and t-SNE. |
| **CLI** | `parallel_complexity_cli.py --missing-values {drop_rows,fill_zero,impute_median,impute_mean}` (default: `impute_median`). |

---

## Project Pages

- `🧮 Complexity Calculator`
  - Analyze one dataset
  - Choose `pycol`, `pymfe`, or both
  - Choose **missing-value** strategy for features
  - Select all metrics or a custom subset
  - Download one-row complexity CSV
  - Visualize t-SNE

- `📊 Dataset Comparison`
  - Add multiple datasets to a comparison list
  - Choose one **missing-value** strategy shared by all listed datasets
  - Compute selected metrics across all datasets
  - Download comparison CSV
  - View grouped bar chart for selected metrics

- `📚 Metric Reference`
  - Review available metrics
  - See descriptions and references

---

## Requirements

- Python 3.10+ (tested on Python 3.13)
- `pip`

Install dependencies from:

- `requirements.txt`

---

## Quick Start

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Stop environment later with:

```bash
deactivate
```

---

## How To Use

## 1) Complexity Calculator

1. Open `🧮 Complexity Calculator`.
2. Choose dataset source:
   - `Upload CSV`
   - `UCI`
   - `OpenML`
3. For UCI/OpenML you can choose input mode:
   - `Use ID` (example: `53` for UCI Iris, `61` for OpenML Iris)
   - `Use Link` (example: `https://archive.ics.uci.edu/dataset/53/iris`)
4. After loading:
   - review **Dataset summary**
   - choose label/target column
   - choose **missing values** handling for features (after encoding): drop rows, fill zero, impute median/mean (see section below)
5. Select complexity libraries and metrics:
   - `pycol`, `pymfe`, or both
   - all metrics or selected metrics
6. Click `Compute complexity`.
7. Download CSV (file name defaults to dataset name + `_complexity.csv`).
8. Optionally click `Show t-SNE of dataset`.

---

## 2) Dataset Comparison

1. Open `📊 Dataset Comparison`.
2. Add datasets one by one:
   - select source
   - load dataset
   - choose label column
   - click `Add dataset to comparison list`
3. Choose **missing values** strategy (applies to all listed datasets for metrics and t-SNE).
4. Choose libraries and metrics.
5. Click `Compute comparison metrics`.
6. Review table and download `datasets_complexity_comparison.csv`.
7. Select metrics in the chart section to see grouped bar comparison:
   - x-axis: metrics
   - y-axis: values
   - legend: dataset names

---

## 3) Parallel CLI (Single Dataset)

### What it is

`parallel_complexity_cli.py` is a command-line tool for **one dataset** that computes complexity metrics in parallel across CPU cores.

This is useful when:

- you run on a server/HPC machine
- metrics are time-consuming
- you want faster turnaround than sequential computation
- you want a reproducible non-UI pipeline run

### What arguments it asks for

- **Source**: `csv`, `uci`, or `openml`
- **Dataset reference** (`--ref`):
  - CSV path (for `csv`)
  - For `uci` / `openml`: **either** a plain numeric id (e.g. `53`, `61`) **or** a full dataset URL. The CLI finds the dataset id inside the string (same idea as the Streamlit pages).
- **Metrics**: `all` or comma-separated list
- **CPU cores**: `--n-jobs`
- **Label column**: target/label column name
- **Missing values** (`--missing-values`): `drop_rows`, `fill_zero`, `impute_median` (default), or `impute_mean`

### Core behavior

- Runs selected metric tasks in parallel using multiprocessing
- Supports `pycol`, `pymfe`, or `both`
- **Progress feedback** (when `stderr` is a TTY): numbered phases (load → encode → each library → save) plus a **per-metric progress bar** (`tqdm` on `stderr`). Install deps with `pip install -r requirements.txt` (includes `tqdm`). Use **`--no-progress`** for silent, machine-friendly logs.
- Merges all computed metrics into one final CSV row
- Keeps metric name prefixes:
  - `pycol_*`
  - `pymfe_*`

### CLI examples

#### A) CSV + both libraries + all metrics + 8 cores

```bash
python3 parallel_complexity_cli.py \
  --source csv \
  --ref "/path/to/your_dataset.csv" \
  --label-column target \
  --library both \
  --pycol-metrics all \
  --pymfe-metrics all \
  --n-jobs 8 \
  --missing-values impute_median \
  --output-csv results/one_dataset_parallel.csv
```

#### B) UCI (by link) + PyCol + selected metrics

```bash
python3 parallel_complexity_cli.py \
  --source uci \
  --ref "https://archive.ics.uci.edu/dataset/53/iris" \
  --label-column target \
  --library pycol \
  --metrics "F1,N1,N2,N3" \
  --n-jobs 4 \
  --output-csv results/uci_iris_pycol.csv
```

#### C) OpenML (by id or link) + PyMFE + all metrics

```bash
python3 parallel_complexity_cli.py \
  --source openml \
  --ref "61" \
  --label-column target \
  --library pymfe \
  --metrics all \
  --n-jobs 6 \
  --output-csv results/openml61_pymfe.csv
```

Same with a link instead of id:

```bash
python3 parallel_complexity_cli.py \
  --source openml \
  --ref "https://www.openml.org/d/61" \
  --label-column target \
  --library pymfe \
  --metrics all \
  --n-jobs 6 \
  --output-csv results/openml61_pymfe.csv
```

### Notes

- For `--library pycol` or `--library pymfe`, use `--metrics`.
- For `--library both`, use:
  - `--pycol-metrics`
  - `--pymfe-metrics`
- If you pass a UCI/OpenML link, the script extracts the dataset ID automatically.

---

## 4) Batch run (many datasets, one after another)

Edit **`run_batch_parallel.sh`** in the repo root:

- At the top, set shared **`parallel_complexity_cli.py`** options (`LIBRARY`, `N_JOBS`, `MISSING_VALUES`, `OUTPUT_CSV`, metric lists, `NO_PROGRESS`, `CONTINUE_ON_ERROR`, `DRY_RUN`).
- In the **`DATASETS=( ... )`** array, add one quoted string per dataset:  
  **`source|ref|label_column`**  
  - `source`: `uci`, `openml`, or `csv`  
  - `ref`: full UCI/OpenML URL (or id), or **absolute path** to a CSV when `source` is `csv`  
  - `label_column`: name of the target column in the dataframe the CLI sees (`target` for UCI/OpenML loaded by this CLI; your real column name for your own CSV)

The script runs **`parallel_complexity_cli.py` once per entry, in order** (each run still uses `N_JOBS` internally). One **`OUTPUT_CSV`** accumulates rows (CLI **upserts** by `dataset_name`).

```bash
cd /path/to/DataComplexity
chmod +x run_batch_parallel.sh
./run_batch_parallel.sh
```

Set `DRY_RUN=1` at the top of the script to print commands without running.

---

## Metric Naming Convention

To avoid confusion, metric columns are prefixed by source library:

- `pycol_*` -> metrics computed by PyCol
- `pymfe_*` -> metrics computed by PyMFE

---

## Troubleshooting

- If page changes do not appear, refresh browser tab.
- If import errors occur, ensure virtual environment is active and reinstall:

```bash
pip install -r requirements.txt
```

- If a UCI/OpenML dataset fails to load:
  - check ID/link format
  - ensure internet connection
  - verify that dataset has target/label information

- **`drop_rows` leaves no rows (e.g. UCI Adult):**  
  Adult has missing values in several features ([UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)). With `drop_rows`, only rows with **no** NaN in **any** feature survive, which can delete the whole table. Use **`impute_median`** or **`impute_mean`** for this dataset, or keep `drop_rows` only if you accept a much smaller complete-case subset (the app now drops all-NaN feature columns first, then listwise row deletion).

---

## Maintainer / Group

- **Maintainer:** Aghil Hooshmand  
- **Role:** Research Fellow, FORGE project (Federated Offline Reflection Grammatical Evolution)  
- **Group:** Biocomputing and Development Systems (BDS), University of Limerick  
- **Department:** Computer Science and Information Systems (CSIS)  
- **Institute:** Lero, the Irish Software Engineering Research Institute  
- **Contact:** `aghil.hooshmand@ul.ie`

