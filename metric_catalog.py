from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDoc:
    key: str
    title: str
    description: str
    reference: str


PYCOL_METRICS: dict[str, MetricDoc] = {
    "F1": MetricDoc("F1", "F1 (Maximum Fisher's discriminant ratio)", "Feature overlap measure based on Fisher's discriminant idea.", "Lorena et al. 2019; pycol-complexity docs"),
    "F1v": MetricDoc("F1v", "F1v (Directional-vector Fisher ratio)", "Directional variant of F1 considering feature interactions.", "Lorena et al. 2019; pycol-complexity docs"),
    "F2": MetricDoc("F2", "F2 (Volume of overlap region)", "Measures overlap of class feature ranges.", "Lorena et al. 2019; pycol-complexity docs"),
    "F3": MetricDoc("F3", "F3 (Feature efficiency)", "Fraction of samples separable by best individual feature.", "Lorena et al. 2019; pycol-complexity docs"),
    "F4": MetricDoc("F4", "F4 (Collective feature efficiency)", "Feature-set version of efficiency after iterative exclusion.", "Lorena et al. 2019; pycol-complexity docs"),
    "N1": MetricDoc("N1", "N1 (Borderline points via MST)", "Fraction of points near class boundary from spanning tree structure.", "Lorena et al. 2019; pycol-complexity docs"),
    "N2": MetricDoc("N2", "N2 (Intra/extra class NN ratio)", "Ratio comparing nearest same-class and nearest enemy distances.", "Lorena et al. 2019; pycol-complexity docs"),
    "N3": MetricDoc("N3", "N3 (1-NN leave-one-out error)", "Local neighborhood ambiguity measured with nearest-neighbor error.", "Lorena et al. 2019; pycol-complexity docs"),
    "N4": MetricDoc("N4", "N4 (Non-linearity of 1-NN)", "Nearest-neighbor non-linearity through interpolation behavior.", "Lorena et al. 2019; pycol-complexity docs"),
    "T1": MetricDoc("T1", "T1 (Hypersphere coverage)", "Topological complexity from class adherence subsets.", "Lorena et al. 2019; pycol-complexity docs"),
    "C1": MetricDoc("C1", "C1 (Class entropy)", "Imbalance-related complexity from class entropy.", "Lorena et al. 2019; pycol-complexity docs"),
    "C2": MetricDoc("C2", "C2 (Imbalance ratio)", "Multi-class imbalance ratio complexity.", "Lorena et al. 2019; pycol-complexity docs"),
    "LSC": MetricDoc("LSC", "LSC (Local set cardinality)", "Neighborhood set-cardinality complexity.", "Lorena et al. 2019; pycol-complexity docs"),
}


PYMFE_COMPLEXITY_METRICS: dict[str, MetricDoc] = {
    "f1": MetricDoc("f1", "f1 (Fisher ratio)", "Fisher-style overlap/separability metric family.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "f1v": MetricDoc("f1v", "f1v (Directional Fisher)", "Directional Fisher separability meta-feature.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "f2": MetricDoc("f2", "f2 (Range overlap)", "Class range overlap indicator.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "f3": MetricDoc("f3", "f3 (Feature efficiency)", "Single-feature efficiency against overlap.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "f4": MetricDoc("f4", "f4 (Collective efficiency)", "Feature-set efficiency against overlap.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "l1": MetricDoc("l1", "l1 (Linear separability error)", "Hinge-loss style linear complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "l2": MetricDoc("l2", "l2 (Linear classifier error)", "Linear classifier training error complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "l3": MetricDoc("l3", "l3 (Non-linearity of linear classifier)", "Interpolation-based non-linearity of linear models.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "n1": MetricDoc("n1", "n1 (MST boundary)", "Boundary complexity from graph edges crossing classes.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "n2": MetricDoc("n2", "n2 (Intra/extra NN ratio)", "Local distance-ratio complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "n3": MetricDoc("n3", "n3 (1-NN error)", "Neighborhood ambiguity by nearest-neighbor error.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "n4": MetricDoc("n4", "n4 (1-NN non-linearity)", "Interpolation-based non-linearity for 1-NN.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "t1": MetricDoc("t1", "t1 (Hypersphere cover)", "Topological complexity from local coverage.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "t2": MetricDoc("t2", "t2 (Features per instance)", "Dimensional sparsity indicator.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "t3": MetricDoc("t3", "t3 (PCA dimensions per instance)", "Intrinsic dimensionality proxy.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "t4": MetricDoc("t4", "t4 (PCA/original ratio)", "Dimensionality concentration complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "c1": MetricDoc("c1", "c1 (Class entropy)", "Class-distribution entropy complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "c2": MetricDoc("c2", "c2 (Imbalance ratio)", "Class imbalance ratio complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "cls_coef": MetricDoc("cls_coef", "cls_coef (Clustering coefficient)", "Graph-based clustering complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "density": MetricDoc("density", "density (Graph density)", "Graph density complexity in neighborhood representation.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "hubs": MetricDoc("hubs", "hubs (Hub score)", "Hubness/eigenvector-centrality complexity.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
    "lsc": MetricDoc("lsc", "lsc (Local set cardinality)", "Local set complexity indicator.", "https://pymfe.readthedocs.io/en/latest/generated/pymfe.complexity.MFEComplexity.html"),
}

