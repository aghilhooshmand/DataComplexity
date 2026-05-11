# Dataset Complexity App

A Streamlit application for dataset complexity analysis using:

- `pycol-complexity`
- `pymfe` (complexity group)

The app supports:

- single-dataset analysis
- multi-dataset comparison
- dataset loading from CSV upload, UCI, and OpenML
- t-SNE visualization
- CSV export for results
- parallel CLI execution for one dataset (speedup on multi-core servers)

---

## Project Pages

- `🧮 Complexity Calculator`
  - Analyze one dataset
  - Choose `pycol`, `pymfe`, or both
  - Select all metrics or a custom subset
  - Download one-row complexity CSV
  - Visualize t-SNE

- `📊 Dataset Comparison`
  - Add multiple datasets to a comparison list
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
3. Choose libraries and metrics.
4. Click `Compute comparison metrics`.
5. Review table and download `datasets_complexity_comparison.csv`.
6. Select metrics in the chart section to see grouped bar comparison:
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
- **Dataset reference**:
  - CSV path (for `csv`)
  - ID or link (for `uci` / `openml`)
- **Metrics**: `all` or comma-separated list
- **CPU cores**: `--n-jobs`
- **Label column**: target/label column name

### Core behavior

- Runs selected metric tasks in parallel using multiprocessing
- Supports `pycol`, `pymfe`, or `both`
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

#### C) OpenML (by id) + PyMFE + all metrics

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

### Notes

- For `--library pycol` or `--library pymfe`, use `--metrics`.
- For `--library both`, use:
  - `--pycol-metrics`
  - `--pymfe-metrics`
- If you pass a UCI/OpenML link, the script extracts the dataset ID automatically.

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

---

## Maintainer / Group

- **Maintainer:** Aghil Hooshmand  
- **Role:** Research Fellow, FORGE project (Federated Offline Reflection Grammatical Evolution)  
- **Group:** Biocomputing and Development Systems (BDS), University of Limerick  
- **Department:** Computer Science and Information Systems (CSIS)  
- **Institute:** Lero, the Irish Software Engineering Research Institute  
- **Contact:** `aghil.hooshmand@ul.ie`

