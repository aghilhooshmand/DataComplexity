from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricDirection = Literal["higher", "lower", "context"]

METRIC_DIRECTION_LABELS: dict[MetricDirection, str] = {
    "higher": "↑ higher = harder",
    "lower": "↓ higher = easier",
    "context": "~ context-dependent",
}


def metric_direction_label(direction: MetricDirection) -> str:
    return METRIC_DIRECTION_LABELS[direction]


@dataclass(frozen=True)
class MetricDoc:
    key: str
    title: str
    description: str
    reference: str
    direction: MetricDirection = "higher"


PYCOL_METRICS: dict[str, MetricDoc] = {
    # Feature overlap
    "F1": MetricDoc(
        "F1",
        "F1 (Maximum Fisher's discriminant ratio)",
        "Feature overlap via Fisher's discriminant (PyCol uses 1/(1+ratio), so larger values mean less separation).",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "F1v": MetricDoc(
        "F1v",
        "F1v (Directional-vector Fisher ratio)",
        "Directional Fisher ratio with feature interactions (same 1/(1+ratio) transform as F1 in PyCol).",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "F2": MetricDoc(
        "F2",
        "F2 (Volume of overlap region)",
        "Normalized volume where class feature ranges overlap (one-vs-one for multi-class).",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "F3": MetricDoc(
        "F3",
        "F3 (Feature efficiency)",
        "Minimum fraction of points lying in the pairwise feature overlap region.",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "F4": MetricDoc(
        "F4",
        "F4 (Collective feature efficiency)",
        "Iterative feature-removal version of F3; fraction still in overlap after greedy exclusion.",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "input_noise": MetricDoc(
        "input_noise",
        "input_noise (Input noise)",
        "Share of feature values from one class that fall inside another class's range.",
        "Van der Walt & Barnard 2007; pycol-complexity docs",
        "higher",
    ),
    # Instance overlap
    "R_value": MetricDoc(
        "R_value",
        "R-value (Imbalance-aware overlap)",
        "Pairwise overlap between classes adjusted for class imbalance.",
        "Borsos et al. 2018; pycol-complexity docs",
        "higher",
    ),
    "deg_overlap": MetricDoc(
        "deg_overlap",
        "deg_overlap (Degree of overlap)",
        "Fraction of points whose k nearest neighbours include at least one other class.",
        "Mercier et al. 2018; pycol-complexity docs",
        "higher",
    ),
    "N3": MetricDoc(
        "N3",
        "N3 (1-NN leave-one-out error)",
        "Fraction of points misclassified by leave-one-out 1-NN.",
        "Ho & Basu 2002; pycol-complexity docs",
        "higher",
    ),
    "SI": MetricDoc(
        "SI",
        "SI (Separability index)",
        "Fraction of points whose nearest neighbours are predominantly same-class.",
        "Thornton 1998; pycol-complexity docs",
        "lower",
    ),
    "N4": MetricDoc(
        "N4",
        "N4 (Non-linearity of 1-NN)",
        "1-NN error on synthetically interpolated points (non-linearity proxy).",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "kDN": MetricDoc(
        "kDN",
        "kDN (k-disagreeing neighbours)",
        "Average fraction of k nearest neighbours with a different class label.",
        "Smith et al. 2014; pycol-complexity docs",
        "higher",
    ),
    "D3_value": MetricDoc(
        "D3_value",
        "D3 (Class density in overlap)",
        "Per-class count of points in ambiguous neighbourhoods (returned as vector; app averages).",
        "Sotoca et al. 2006; pycol-complexity docs",
        "higher",
    ),
    "CM": MetricDoc(
        "CM",
        "CM (Complexity metric, kNN)",
        "Fraction of instances with more than half of k neighbours from other classes.",
        "Anwar et al. 2014; pycol-complexity docs",
        "higher",
    ),
    "borderline": MetricDoc(
        "borderline",
        "borderline (Borderline examples)",
        "Percentage of points with at least two nearest neighbours from other classes.",
        "Napierala et al. 2010; pycol-complexity docs",
        "higher",
    ),
    # Structural overlap
    "N1": MetricDoc(
        "N1",
        "N1 (Borderline points via MST)",
        "Fraction of MST vertices on edges connecting different classes.",
        "Ho & Basu 2002; pycol-complexity docs",
        "higher",
    ),
    "N2": MetricDoc(
        "N2",
        "N2 (Intra/extra class NN ratio)",
        "Ratio of summed same-class vs enemy nearest-neighbour distances (mapped to [0, 1]).",
        "Ho & Basu 2002; pycol-complexity docs",
        "higher",
    ),
    "T1": MetricDoc(
        "T1",
        "T1 (Hypersphere coverage)",
        "Fraction of points that require a non-empty covering hypersphere.",
        "Lorena et al. 2019; pycol-complexity docs",
        "higher",
    ),
    "Clust": MetricDoc(
        "Clust",
        "Clust (Number of clusters)",
        "Local-set cluster cores per instance (fragmentation of class manifolds).",
        "Leyva et al. 2014; pycol-complexity docs",
        "higher",
    ),
    "ONB": MetricDoc(
        "ONB",
        "ONB (Overlap number of balls)",
        "Overlap among class-covering hyperspheres (topology).",
        "Pascual-Triana; pycol-complexity docs",
        "higher",
    ),
    "LSC": MetricDoc(
        "LSC",
        "LSC (Local set cardinality)",
        "Transform of average same-class points within the enemy-distance radius.",
        "Leyva et al. 2014; pycol-complexity docs",
        "higher",
    ),
    "DBC": MetricDoc(
        "DBC",
        "DBC (Decision boundary complexity)",
        "MST edges between hypersphere centers that cross class labels.",
        "Van der Walt et al. 2008; pycol-complexity docs",
        "higher",
    ),
    "NSG": MetricDoc(
        "NSG",
        "NSG (Samples per group)",
        "Average number of samples per covering hypersphere group.",
        "Van der Walt & Barnard 2007; pycol-complexity docs",
        "context",
    ),
    "ICSV": MetricDoc(
        "ICSV",
        "ICSV (Inter-class scale variation)",
        "Standard deviation of hypersphere densities across classes.",
        "Van der Walt & Barnard 2007; pycol-complexity docs",
        "higher",
    ),
    # Multiresolution overlap
    "MRCA": MetricDoc(
        "MRCA",
        "MRCA (Multiresolution complexity analysis)",
        "Multiresolution index from hypersphere profiles clustered in scale space.",
        "Armano & Tamponi 2016; pycol-complexity docs",
        "higher",
    ),
    "C1": MetricDoc(
        "C1",
        "C1 (Case-base complexity profile)",
        "1 minus average enemy fraction in multi-scale hyperspheres (not class entropy).",
        "Massie et al. 2005; pycol-complexity docs",
        "lower",
    ),
    "C2": MetricDoc(
        "C2",
        "C2 (Similarity-weighted case-base complexity)",
        "Similarity-weighted variant of C1 (not imbalance ratio).",
        "Cummins 2013; pycol-complexity docs",
        "lower",
    ),
    "purity": MetricDoc(
        "purity",
        "purity (Grid purity)",
        "Multiresolution grid-cell homogeneity (higher = purer cells).",
        "Singh 2003; pycol-complexity docs",
        "lower",
    ),
    "neighbourhood_separability": MetricDoc(
        "neighbourhood_separability",
        "neighbourhood_separability",
        "Multiresolution AUC of same-class neighbour proportions (higher = more separable).",
        "Singh 2003; pycol-complexity docs",
        "lower",
    ),
}


PYMFE_COMPLEXITY_METRICS: dict[str, MetricDoc] = {
    "f1": MetricDoc(
        "f1",
        "f1 (Fisher ratio)",
        "Fisher discriminant ratio between classes (higher = more separable).",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "lower",
    ),
    "f1v": MetricDoc(
        "f1v",
        "f1v (Directional Fisher)",
        "Directional Fisher separability meta-feature.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "lower",
    ),
    "f2": MetricDoc(
        "f2",
        "f2 (Range overlap)",
        "Class range overlap indicator.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "f3": MetricDoc(
        "f3",
        "f3 (Feature efficiency)",
        "Single-feature separability efficiency.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "lower",
    ),
    "f4": MetricDoc(
        "f4",
        "f4 (Collective efficiency)",
        "Feature-set efficiency after iterative exclusion.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "lower",
    ),
    "l1": MetricDoc(
        "l1",
        "l1 (Linear separability error)",
        "Hinge-loss style linear separability error.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "l2": MetricDoc(
        "l2",
        "l2 (Linear classifier error)",
        "Linear classifier training error.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "l3": MetricDoc(
        "l3",
        "l3 (Non-linearity of linear classifier)",
        "Interpolation-based non-linearity of linear models.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "n1": MetricDoc(
        "n1",
        "n1 (MST boundary)",
        "Boundary complexity from MST edges crossing classes.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "n2": MetricDoc(
        "n2",
        "n2 (Intra/extra NN ratio)",
        "Local distance-ratio complexity.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "n3": MetricDoc(
        "n3",
        "n3 (1-NN error)",
        "Neighborhood ambiguity by nearest-neighbor error.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "n4": MetricDoc(
        "n4",
        "n4 (1-NN non-linearity)",
        "Interpolation-based non-linearity for 1-NN.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "t1": MetricDoc(
        "t1",
        "t1 (Hypersphere cover)",
        "Topological complexity from local hypersphere coverage.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "t2": MetricDoc(
        "t2",
        "t2 (Features per instance)",
        "Ratio of features to instances (dimensionality pressure).",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "t3": MetricDoc(
        "t3",
        "t3 (PCA dimensions per instance)",
        "PCA-based intrinsic dimensionality per instance.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "t4": MetricDoc(
        "t4",
        "t4 (PCA/original ratio)",
        "Ratio of PCA dimensions retained vs original feature count.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "context",
    ),
    "c1": MetricDoc(
        "c1",
        "c1 (Class entropy)",
        "Shannon entropy of the class distribution (balance, not geometric overlap).",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "context",
    ),
    "c2": MetricDoc(
        "c2",
        "c2 (Imbalance ratio)",
        "Class imbalance ratio (minority/majority).",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "cls_coef": MetricDoc(
        "cls_coef",
        "cls_coef (Clustering coefficient)",
        "Graph clustering coefficient in the neighbourhood graph.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "density": MetricDoc(
        "density",
        "density (Graph density)",
        "Graph density in the neighbourhood representation.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "hubs": MetricDoc(
        "hubs",
        "hubs (Hub score)",
        "Hubness / eigenvector-centrality in the neighbourhood graph.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
    "lsc": MetricDoc(
        "lsc",
        "lsc (Local set cardinality)",
        "Local set cardinality complexity indicator.",
        "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html",
        "higher",
    ),
}
