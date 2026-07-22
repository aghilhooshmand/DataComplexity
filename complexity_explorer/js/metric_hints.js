/* Auto-synced from metric_catalog.py — do not edit by hand */
const METRIC_HINTS = {
  "F1": {
    "title": "F1 (Maximum Fisher's discriminant ratio)",
    "blurb": "Feature overlap via Fisher's discriminant (PyCol uses 1/(1+ratio), so larger values mean less separation). PyCol stores 1/(1+Fisher), so higher means worse separation.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "F1v": {
    "title": "F1v (Directional-vector Fisher ratio)",
    "blurb": "Directional Fisher ratio with feature interactions (same 1/(1+ratio) transform as F1 in PyCol). Same 1/(1+ratio) style as F1 in PyCol.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "F2": {
    "title": "F2 (Volume of overlap region)",
    "blurb": "Normalized volume where class feature ranges overlap (one-vs-one for multi-class).",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "F3": {
    "title": "F3 (Feature efficiency)",
    "blurb": "Minimum fraction of points lying in the pairwise feature overlap region.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "F4": {
    "title": "F4 (Collective feature efficiency)",
    "blurb": "Iterative feature-removal version of F3; fraction still in overlap after greedy exclusion.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "input_noise": {
    "title": "input_noise (Input noise)",
    "blurb": "Share of feature values from one class that fall inside another class's range.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "R_value": {
    "title": "R-value (Imbalance-aware overlap)",
    "blurb": "Pairwise overlap between classes adjusted for class imbalance.",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "deg_overlap": {
    "title": "deg_overlap (Degree of overlap)",
    "blurb": "Fraction of points whose k nearest neighbours include at least one other class.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "N3": {
    "title": "N3 (1-NN leave-one-out error)",
    "blurb": "Fraction of points misclassified by leave-one-out 1-NN.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "SI": {
    "title": "SI (Separability index)",
    "blurb": "Fraction of points whose nearest neighbours are predominantly same-class.",
    "hard": "Higher \u2192 less complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "N4": {
    "title": "N4 (Non-linearity of 1-NN)",
    "blurb": "1-NN error on synthetically interpolated points (non-linearity proxy).",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "kDN": {
    "title": "kDN (k-disagreeing neighbours)",
    "blurb": "Average fraction of k nearest neighbours with a different class label.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "D3_value": {
    "title": "D3 (Class density in overlap)",
    "blurb": "Per-class count of points in ambiguous neighbourhoods (returned as vector; app averages).",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "CM": {
    "title": "CM (Complexity metric, kNN)",
    "blurb": "Fraction of instances with more than half of k neighbours from other classes.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "borderline": {
    "title": "borderline (Borderline examples)",
    "blurb": "Percentage of points with at least two nearest neighbours from other classes.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "N1": {
    "title": "N1 (Borderline points via MST)",
    "blurb": "Fraction of MST vertices on edges connecting different classes.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "N2": {
    "title": "N2 (Intra/extra class NN ratio)",
    "blurb": "Ratio of summed same-class vs enemy nearest-neighbour distances (mapped to [0, 1]).",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "T1": {
    "title": "T1 (Hypersphere coverage)",
    "blurb": "Fraction of points that require a non-empty covering hypersphere.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "Clust": {
    "title": "Clust (Number of clusters)",
    "blurb": "Local-set cluster cores per instance (fragmentation of class manifolds).",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "ONB": {
    "title": "ONB (Overlap number of balls)",
    "blurb": "Overlap among class-covering hyperspheres (topology).",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "LSC": {
    "title": "LSC (Local set cardinality)",
    "blurb": "Transform of average same-class points within the enemy-distance radius. [Likely] In some papers raw LSC is higher=easier; this project's PyCol column is treated as higher=harder after transform.",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "DBC": {
    "title": "DBC (Decision boundary complexity)",
    "blurb": "MST edges between hypersphere centers that cross class labels.",
    "hard": "Higher \u2192 more complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "NSG": {
    "title": "NSG (Samples per group)",
    "blurb": "Average number of samples per covering hypersphere group.",
    "hard": "Context-dependent (not simply high=hard)",
    "certainty": "context",
    "certaintyLabel": "[Context]"
  },
  "ICSV": {
    "title": "ICSV (Inter-class scale variation)",
    "blurb": "Standard deviation of hypersphere densities across classes.",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "MRCA": {
    "title": "MRCA (Multiresolution complexity analysis)",
    "blurb": "Multiresolution index from hypersphere profiles clustered in scale space.",
    "hard": "Higher \u2192 more complex",
    "certainty": "likely",
    "certaintyLabel": "[Likely]"
  },
  "C1": {
    "title": "C1 (Case-base complexity profile)",
    "blurb": "1 minus average enemy fraction in multi-scale hyperspheres (not class entropy).",
    "hard": "Higher \u2192 less complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "C2": {
    "title": "C2 (Similarity-weighted case-base complexity)",
    "blurb": "Similarity-weighted variant of C1 (not imbalance ratio).",
    "hard": "Higher \u2192 less complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "purity": {
    "title": "purity (Grid purity)",
    "blurb": "Multiresolution grid-cell homogeneity (higher = purer cells).",
    "hard": "Higher \u2192 less complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  },
  "neighbourhood_separability": {
    "title": "neighbourhood_separability",
    "blurb": "Multiresolution AUC of same-class neighbour proportions (higher = more separable).",
    "hard": "Higher \u2192 less complex",
    "certainty": "certain",
    "certaintyLabel": "[Certain]"
  }
};
