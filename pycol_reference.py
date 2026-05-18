"""PyCol doc bundle: overlap taxonomy, figures, and sample datasets (from pycol-doc/)."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PYCOL_DOC_ROOT = Path(__file__).resolve().parent / "pycol-doc"
PYCOL_DATASET_DIR = PYCOL_DOC_ROOT / "dataset"
PYCOL_IMG_DIR = PYCOL_DOC_ROOT / "img"

PYCOL_GITHUB_README = "https://github.com/miriamspsantos/pycol/blob/doc/README.md"
PYCOL_GITHUB_IMG = "https://github.com/miriamspsantos/pycol/tree/doc/img"
PYCOL_AIRE_PAPER = "https://miriamspsantos.github.io/pdf-files/AIRE_2022.pdf"


@dataclass(frozen=True)
class PycolOverlapCategory:
    key: str
    title: str
    measures: tuple[str, ...]


PYCOL_OVERLAP_CATEGORIES: tuple[PycolOverlapCategory, ...] = (
    PycolOverlapCategory(
        "feature",
        "Feature overlap",
        ("F1", "F1v", "F2", "F3", "F4", "input_noise (IN)"),
    ),
    PycolOverlapCategory(
        "instance",
        "Instance overlap",
        (
            "R-value",
            "Raug",
            "degOver",
            "N3",
            "SI",
            "N4",
            "kDN",
            "D3",
            "CM",
            "wCM",
            "dwCM",
            "borderline",
            "IPoints",
        ),
    ),
    PycolOverlapCategory(
        "structural",
        "Structural overlap",
        ("N1", "T1", "Clust", "ONB", "LSC", "DBC", "N2", "NSG", "ICSV"),
    ),
    PycolOverlapCategory(
        "multiresolution",
        "Multiresolution overlap",
        ("MRCA", "C1", "C2", "purity", "neighbourhood_separability"),
    ),
)


@dataclass(frozen=True)
class PycolFigure:
    file: str
    title: str
    category: str
    caption: str = ""


PYCOL_FIGURES: tuple[PycolFigure, ...] = (
    PycolFigure("taxonomy_class_overlap.pdf", "Class overlap taxonomy", "Overview"),
    PycolFigure("class_overlap_approaches.pdf", "Class overlap approaches", "Overview"),
    PycolFigure("data_typology.pdf", "Data typology", "Overview"),
    PycolFigure("domains.pdf", "Domains", "Overview"),
    PycolFigure("F1.pdf", "F1 — maximum Fisher discriminant ratio", "Feature overlap"),
    PycolFigure("F1_F2.pdf", "F1 and F2", "Feature overlap"),
    PycolFigure("F3.pdf", "F3 — maximum individual feature efficiency", "Feature overlap"),
    PycolFigure("F4.pdf", "F4 — collective feature efficiency", "Feature overlap"),
    PycolFigure("r_value.pdf", "R-value", "Instance overlap"),
    PycolFigure("degOver.pdf", "degOver", "Instance overlap"),
    PycolFigure("invasive_points.pdf", "Invasive points (IPoints)", "Instance overlap"),
    PycolFigure("N4.pdf", "N4 — 1-NN non-linearity", "Instance overlap"),
    PycolFigure("N1.pdf", "N1 — borderline points", "Structural overlap"),
    PycolFigure("N2.pdf", "N2 — intra/extra class NN distance ratio", "Structural overlap"),
    PycolFigure("DBC.pdf", "DBC — decision boundary complexity", "Structural overlap"),
    PycolFigure("ONB.pdf", "ONB — overlap number of balls", "Structural overlap"),
    PycolFigure("LS.pdf", "LSC — local set cardinality", "Structural overlap"),
    PycolFigure("T1_dcol.pdf", "T1 (DCoL view)", "Structural overlap"),
    PycolFigure("T1_ecol.pdf", "T1 (ECoL view)", "Structural overlap"),
    PycolFigure("number_of_clusters.pdf", "Number of clusters (Clust)", "Structural overlap"),
    PycolFigure("MRCA.pdf", "MRCA — multiresolution complexity", "Multiresolution overlap"),
    PycolFigure("C1.pdf", "C1 — case-base complexity profile", "Multiresolution overlap"),
    PycolFigure("purity.pdf", "Purity", "Multiresolution overlap"),
    PycolFigure("NC_SOL_1.pdf", "Neighbourhood separability (1)", "Multiresolution overlap"),
    PycolFigure("NC_SOL_2.pdf", "Neighbourhood separability (2)", "Multiresolution overlap"),
)


@dataclass(frozen=True)
class PycolSampleDataset:
    file: str
    title: str
    notes: str
    label_column: str = "class"


PYCOL_SAMPLE_DATASETS: tuple[PycolSampleDataset, ...] = (
    PycolSampleDataset("61_iris.arff", "Iris (3 classes)", "Classic multi-class; UCI 61.", "class"),
    PycolSampleDataset("61_iris_binary.arff", "Iris binary", "Two-class variant.", "class"),
    PycolSampleDataset("ecoli.dat", "E. coli", "8 classes; protein localization.", "Class"),
    PycolSampleDataset("balance.dat", "Balance scale", "3 classes.", "Class"),
    PycolSampleDataset("yeast.data", "Yeast", "10 classes; gene expression (no ARFF header).", "target"),
    PycolSampleDataset("abalone.data", "Abalone", "Regression target (rings); 9 attributes.", "Rings"),
    PycolSampleDataset("car.data", "Car evaluation", "Multi-class (nominal CSV).", "class"),
    PycolSampleDataset("coil2000.dat", "COIL 2000", "Marketing response.", "class"),
    PycolSampleDataset("newthyroid.dat", "New thyroid", "3 classes.", "class"),
    PycolSampleDataset("titanic.dat", "Titanic", "Survival.", "class"),
    PycolSampleDataset("circles-2d.arff", "Circles 2D", "Synthetic non-linear.", "class"),
    PycolSampleDataset("sphere-2d-3000-u.arff", "Sphere 2D", "Synthetic.", "class"),
    PycolSampleDataset("flower-3d.arff", "Flower 3D", "Synthetic 3D.", "class"),
    PycolSampleDataset("paw3-2d-learn-1.arff", "PAW 3D 2D projection", "Synthetic overlap.", "class"),
    PycolSampleDataset("paw3-3d.arff", "PAW 3D", "Synthetic 3D.", "class"),
    PycolSampleDataset("test1.arff", "Test 1", "Synthetic benchmark.", "class"),
    PycolSampleDataset("test2.arff", "Test 2", "Synthetic benchmark.", "class"),
    PycolSampleDataset("test3.arff", "Test 3", "Synthetic benchmark.", "class"),
    PycolSampleDataset("test4.arff", "Test 4", "Synthetic benchmark.", "class"),
    PycolSampleDataset("t1_test.arff", "T1 test", "T1 complexity toy set.", "class"),
)


def _decode_arff_nominals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object and len(out) and isinstance(out[c].iloc[0], bytes):
            out[c] = out[c].str.decode("utf-8", errors="ignore")
    return out


def _load_arff_like(path: Path) -> pd.DataFrame:
    from scipy.io import arff

    data, _meta = arff.loadarff(path)
    return _decode_arff_nominals(pd.DataFrame(data))


def _load_yeast(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    raw.columns = [f"f{i}" for i in range(raw.shape[1] - 1)] + ["target"]
    return raw


def _load_abalone(path: Path) -> pd.DataFrame:
    names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
        "Rings",
    ]
    return pd.read_csv(path, header=None, names=names)


def _load_car(path: Path) -> pd.DataFrame:
    names = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class",
    ]
    return pd.read_csv(path, header=None, names=names)


def load_pycol_sample_dataset(spec: PycolSampleDataset) -> pd.DataFrame:
    path = PYCOL_DATASET_DIR / spec.file
    if not path.is_file():
        raise FileNotFoundError(f"Sample dataset not found: {path}")

    if spec.file.endswith(".data") and spec.file == "yeast.data":
        return _load_yeast(path)
    if spec.file.endswith(".data") and spec.file == "abalone.data":
        return _load_abalone(path)
    if spec.file == "car.data":
        return _load_car(path)

    df = _load_arff_like(path)
    if spec.label_column not in df.columns:
        last = df.columns[-1]
        df = df.rename(columns={last: spec.label_column})
    return df


def pycol_sample_dataset_name(spec: PycolSampleDataset) -> str:
    stem = Path(spec.file).stem
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._") or "pycol_sample"


def figure_path(fig: PycolFigure) -> Path:
    return PYCOL_IMG_DIR / fig.file


def render_pdf_embed(path: Path, *, height: int = 520) -> None:
    """Embed a local PDF in Streamlit (browser iframe)."""
    import streamlit as st

    if not path.is_file():
        st.warning(f"Figure not found: `{path}`")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="{height}" style="border:1px solid #e2e8f0;border-radius:8px;"></iframe>',
        unsafe_allow_html=True,
    )


def list_available_sample_datasets() -> list[PycolSampleDataset]:
    return [s for s in PYCOL_SAMPLE_DATASETS if (PYCOL_DATASET_DIR / s.file).is_file()]


def list_available_figures() -> list[PycolFigure]:
    return [f for f in PYCOL_FIGURES if figure_path(f).is_file()]
