"""Deterministic initializations for SG-t-SNE-Π.

PCA-init removes seed-driven variation so the true λ-effect signal becomes
measurable without the random-init noise floor. Convention follows
Kobak & Linderman 2021 (Nature Biotech) / openTSNE: TruncatedSVD on the
input, center, rescale max-abs to a small value (default 1e-4) so the
optimization can move freely in the early-exaggeration phase.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


def pca_init(
    adj: sp.spmatrix,
    d: int = 2,
    scale: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """Return a deterministic (n, d) float64 PCA initialization for sgtsnepi.

    Operates on the sparse adjacency directly (no dense conversion). Centers
    the embedding and rescales so max(|Y|) == scale.
    """
    svd = TruncatedSVD(n_components=d, random_state=random_state)
    Y = svd.fit_transform(adj).astype(np.float64)
    Y -= Y.mean(axis=0, keepdims=True)
    m = float(np.max(np.abs(Y)))
    if m > 0:
        Y *= scale / m
    return Y
