"""ZADU metric helpers for both feature-bearing and graph-only inputs.

Single source of truth used by `scripts/run_e3_baselines.py` and
`scripts/merge_e3_results.py`. For graph-only datasets the input ranking is
computed from rows of the stochastic kNN/adjacency matrix (subsampled to
`max_n` to keep memory bounded).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

METRIC_KEYS = (
    "trustworthiness",
    "continuity",
    "label_trustworthiness",
    "label_continuity",
)


def _empty() -> dict[str, float]:
    return {k: float("nan") for k in METRIC_KEYS}


def _zadu(features: np.ndarray, Y: np.ndarray, labels: np.ndarray | None) -> dict[str, float]:
    from zadu import ZADU
    spec: list[dict] = [{"id": "trustworthiness_continuity", "params": {"k": 15}}]
    if labels is not None:
        spec.append({"id": "label_trustworthiness_and_continuity", "params": {"cvm": "dsc"}})
    raw = ZADU(spec, features).measure(Y, label=labels)
    out = _empty()
    for entry in raw:
        for k, v in entry.items():
            if k in out:
                out[k] = float(v)
    return out


def compute_metrics(
    features: np.ndarray | None,
    adj: sp.spmatrix | None,
    Y: np.ndarray,
    labels: np.ndarray | None,
    max_n: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute T&C (and Label-T&C if labels given) for either feature- or graph-input.

    For graph-only datasets (`features is None`), uses the rows of the
    stochastic adjacency as the ambient feature representation. Subsamples
    `max_n` nodes (index-aligned across features/adj/Y/labels) to keep
    memory bounded.
    """
    n = Y.shape[0]
    if max_n and n > max_n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_n, replace=False))
    else:
        idx = np.arange(n)

    Y_sub = Y[idx]
    lbl_sub = labels[idx] if labels is not None else None

    if features is not None:
        feat_sub = features[idx]
    elif adj is not None:
        feat_sub = adj[idx].toarray().astype(np.float32)
    else:
        return _empty()

    return _zadu(feat_sub, Y_sub, lbl_sub)
