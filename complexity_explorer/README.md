# Complexity Explorer

Responsive HTML app to browse PyCol complexity results, compare datasets, and plot distributions.

## Open it

Browsers block local CSV fetch from `file://`. Serve the folder:

```bash
cd complexity_explorer
python3 -m http.server 8765
```

Then open [http://localhost:8765](http://localhost:8765).

## Menu

| Section | What it shows |
|---------|----------------|
| **Overview** | KPIs, rows×features scatter, class-count bars |
| **All datasets** | Searchable table (sizes, cheap/standard fill, download/profile links) |
| **Compare** | Multi-select datasets + metrics, bar chart and value table |
| **Complexity glance** | Composite hardness bars, heatmap, radar (pick metrics / datasets) |
| **Distributions** | Histogram for rows, columns, features, classes, or any `pycol_*` metric |
| **Coverage** | Incomplete cheap/standard lists (MRCA gaps, ONB/DBC) |

## Data

CSV path: `data/datasets_complexity_summary.csv`  
(copied from `../results/datasets_complexity_summary.csv`)

Refresh the copy after new Hive runs:

```bash
cp ../results/datasets_complexity_summary.csv data/datasets_complexity_summary.csv
```
