from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from complexity_core import prepare_xy

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY_TSV = PROJECT_ROOT / "all_summary_stats_PMLB_benchmarck_dataset.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pmlb_DS"
PMLB_DATASETS_BASE = (
    "https://raw.githubusercontent.com/EpistasisLab/penn-ml-benchmarks/master/datasets"
)


def sanitize_dataset_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name).strip())
    return cleaned.strip("._") or "dataset"


def load_summary_table(path: Path | str | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else DEFAULT_SUMMARY_TSV
    return pd.read_csv(p, sep="\t")


def select_benchmark_datasets(
    summary: pd.DataFrame,
    *,
    max_instances: int = 21000,
    task: str = "classification",
    pmlb_names: set[str] | None = None,
) -> pd.DataFrame:
    """Rows from summary TSV that match filters and exist in PMLB."""
    out = summary.copy()
    out = out[out["task"].astype(str).str.lower() == task.lower()]
    out = out[out["n_instances"].astype(int) < int(max_instances)]
    if pmlb_names is not None:
        out = out[out["dataset"].astype(str).isin(pmlb_names)]
    return out.sort_values("dataset").reset_index(drop=True)


def encode_pmlb_dataframe(
    df: pd.DataFrame,
    label_col: str = "target",
    *,
    missing_values: str = "impute_median",
) -> pd.DataFrame:
    """Categorical → numeric features (one-hot) and integer class labels in ``target``."""
    _, y, merged = prepare_xy(df, label_col, missing_values=missing_values)
    encoded = merged.drop(columns=["__target__"], errors="ignore")
    encoded["target"] = y
    return encoded


def output_paths(output_dir: Path | str) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "csv": root,
        "metadata": root / "metadata",
        "manifest": root / "manifest.tsv",
        "status": root / "download_status.json",
        "complexity_cache": root / "complexity_metrics.csv",
    }


def csv_path_for_dataset(output_dir: Path | str, dataset_name: str) -> Path:
    return Path(output_dir) / f"{sanitize_dataset_filename(dataset_name)}.csv"


def metadata_dir_for_dataset(output_dir: Path | str, dataset_name: str) -> Path:
    return Path(output_dir) / "metadata" / sanitize_dataset_filename(dataset_name)


def fetch_remote_text(url: str, timeout: float = 30.0) -> str | None:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except requests.RequestException:
        pass
    return None


def fetch_dataset_metadata_files(
    dataset_name: str,
    dest_dir: Path,
    *,
    timeout: float = 30.0,
) -> dict[str, bool]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = f"{PMLB_DATASETS_BASE}/{dataset_name}"
    saved: dict[str, bool] = {}
    for fname in ("README.md", "metadata.yaml"):
        text = fetch_remote_text(f"{base}/{fname}", timeout=timeout)
        if text is None:
            saved[fname] = False
            continue
        (dest_dir / fname).write_text(text, encoding="utf-8")
        saved[fname] = True
    return saved


def load_cached_readme(metadata_dir: Path) -> str | None:
    path = metadata_dir / "README.md"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def load_cached_metadata_yaml(metadata_dir: Path) -> dict[str, Any] | None:
    path = metadata_dir / "metadata.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except yaml.YAMLError:
        return None


def _esc(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text in {"-", "None yet. See our contributing guide to help us add one."}:
        return '<span class="pmlb-muted">—</span>'
    return html.escape(text)


def clean_readme_markdown(text: str) -> str:
    """Drop PMLB README nav links (Metadata | Summary Statistics, profiling banner)."""
    lines: list[str] = []
    skip_patterns = (
        re.compile(r"^\[Metadata\]\(metadata\.yaml\)", re.I),
        re.compile(r"^\[Summary Statistics\]\(summary_stats\.tsv\)", re.I),
        re.compile(r"^\[\*\*Pandas Profiling Report\*\*\]", re.I),
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if any(p.search(stripped) for p in skip_patterns):
            continue
        if stripped == "|" or re.fullmatch(r"\[Metadata\].*\|.*\[Summary Statistics\].*", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


PMLB_PROFILE_CSS = """
<style>
.pmlb-profile {
  font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  color: #0f172a;
  margin-bottom: 1rem;
}
.pmlb-profile h2 {
  margin: 0 0 0.75rem 0;
  font-size: 1.35rem;
  font-weight: 650;
}
.pmlb-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-bottom: 1rem;
}
.pmlb-chip {
  display: inline-block;
  padding: 0.2rem 0.65rem;
  border-radius: 999px;
  background: #e0f2fe;
  color: #0369a1;
  font-size: 0.82rem;
  font-weight: 600;
}
.pmlb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem;
  margin-bottom: 1rem;
}
.pmlb-card {
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  background: #f8fafc;
}
.pmlb-card h3 {
  margin: 0 0 0.45rem 0;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #64748b;
  font-weight: 700;
}
.pmlb-card p {
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.45;
  word-break: break-word;
}
.pmlb-muted { color: #94a3b8; }
.pmlb-features {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  margin-top: 0.25rem;
}
.pmlb-features th {
  text-align: left;
  padding: 0.45rem 0.55rem;
  background: #f1f5f9;
  border-bottom: 2px solid #e2e8f0;
  color: #475569;
  font-weight: 600;
}
.pmlb-features td {
  padding: 0.4rem 0.55rem;
  border-bottom: 1px solid #e2e8f0;
  vertical-align: top;
}
.pmlb-features tr:nth-child(even) td { background: #fafafa; }
</style>
"""


def metadata_yaml_to_html(meta: dict[str, Any], *, dataset_name: str = "") -> str:
    """Render PMLB metadata.yaml as a readable HTML profile."""
    title = _esc(meta.get("dataset") or dataset_name)
    task = _esc(meta.get("task", ""))
    description = _esc(meta.get("description"))
    source = _esc(meta.get("source"))
    publication = _esc(meta.get("publication"))

    keywords = meta.get("keywords") or []
    kw_html = ""
    if isinstance(keywords, list):
        chips = [_esc(k) for k in keywords if k and str(k).strip() and str(k).strip() != "-"]
        if chips:
            kw_html = '<div class="pmlb-chips">' + "".join(
                f'<span class="pmlb-chip">{k}</span>' for k in chips
            ) + "</div>"

    target = meta.get("target") if isinstance(meta.get("target"), dict) else {}
    target_type = _esc(target.get("type"))
    target_desc = _esc(target.get("description"))
    target_code = _esc(target.get("code"))

    features = meta.get("features") or []
    feat_rows = ""
    if isinstance(features, list):
        for feat in features:
            if not isinstance(feat, dict):
                continue
            name = _esc(feat.get("name", ""))
            if not name or name == "—":
                continue
            feat_rows += (
                "<tr>"
                f"<td><strong>{name}</strong></td>"
                f"<td>{_esc(feat.get('type'))}</td>"
                f"<td>{_esc(feat.get('description'))}</td>"
                f"<td>{_esc(feat.get('code'))}</td>"
                f"<td>{_esc(feat.get('transform'))}</td>"
                "</tr>"
            )

    features_table = ""
    if feat_rows:
        features_table = f"""
<h3 style="margin:1rem 0 0.5rem 0;font-size:0.95rem;color:#334155;">Features</h3>
<table class="pmlb-features">
  <thead>
    <tr><th>Name</th><th>Type</th><th>Description</th><th>Code</th><th>Transform</th></tr>
  </thead>
  <tbody>{feat_rows}</tbody>
</table>
"""

    task_chip = f'<span class="pmlb-chip">{task}</span>' if task and task != "—" else ""

    return f"""{PMLB_PROFILE_CSS}
<div class="pmlb-profile">
  <h2>{title}</h2>
  <div class="pmlb-chips">{task_chip}</div>
  {kw_html}
  <div class="pmlb-grid">
    <div class="pmlb-card"><h3>Description</h3><p>{description}</p></div>
    <div class="pmlb-card"><h3>Source</h3><p>{source}</p></div>
    <div class="pmlb-card"><h3>Publication</h3><p>{publication}</p></div>
  </div>
  <div class="pmlb-grid">
    <div class="pmlb-card"><h3>Target type</h3><p>{target_type}</p></div>
    <div class="pmlb-card"><h3>Target description</h3><p>{target_desc}</p></div>
    <div class="pmlb-card"><h3>Target coding</h3><p>{target_code}</p></div>
  </div>
  {features_table}
</div>
"""


def load_manifest(output_dir: Path | str) -> pd.DataFrame:
    paths = output_paths(output_dir)
    if not paths["manifest"].is_file():
        return pd.DataFrame()
    return pd.read_csv(paths["manifest"], sep="\t")


def list_downloaded_datasets(output_dir: Path | str) -> list[str]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    names = sorted(p.stem for p in root.glob("*.csv") if p.name != "manifest.tsv")
    return names


def load_encoded_dataset(output_dir: Path | str, dataset_name: str) -> pd.DataFrame:
    path = csv_path_for_dataset(output_dir, dataset_name)
    if not path.is_file():
        raise FileNotFoundError(f"PMLB CSV not found: {path}")
    return pd.read_csv(path)


def summary_row_for_dataset(summary: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    row = summary.loc[summary["dataset"].astype(str) == dataset_name]
    if row.empty:
        return {"dataset": dataset_name}
    return row.iloc[0].to_dict()


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    df_row = pd.DataFrame([row])
    if manifest_path.is_file():
        existing = pd.read_csv(manifest_path, sep="\t")
        existing = existing[existing["dataset"].astype(str) != str(row.get("dataset", ""))]
        merged = pd.concat([existing, df_row], ignore_index=True)
    else:
        merged = df_row
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("dataset").to_csv(manifest_path, sep="\t", index=False)


def write_download_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
