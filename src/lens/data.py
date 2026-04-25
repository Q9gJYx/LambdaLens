"""Dataset loaders for Lambda Lens experiments.

Cora-only for Phase 3 sanity; Citeseer/MNIST-kNN/ca-AstroPh added as E1 needs them.
"""
from __future__ import annotations

import pickle
import urllib.request as ur
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PLANETOID_BASE = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with ur.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


def load_planetoid(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (adjacency CSR, dense features, integer labels) for a Planetoid dataset."""
    cache = Path(cache_dir) / name
    objs: dict = {}
    for f in PLANETOID_FILES:
        url = f"{PLANETOID_BASE}/ind.{name}.{f}"
        path = cache / f"ind.{name}.{f}"
        _download(url, path)
        if f == "test.index":
            objs[f] = np.array([int(line) for line in path.read_text().split()])
        else:
            with path.open("rb") as fp:
                objs[f] = pickle.load(fp, encoding="latin1")

    test_idx = objs["test.index"]
    test_idx_sorted = np.sort(test_idx)
    allx, tx = objs["allx"], objs["tx"]
    ally, ty = objs["ally"], objs["ty"]
    n = allx.shape[0] + tx.shape[0]

    features = sp.vstack([allx, tx]).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    features = features.toarray().astype(np.float32)

    labels_full = np.vstack([ally, ty])
    labels_full[test_idx, :] = labels_full[test_idx_sorted, :]
    labels = labels_full.argmax(axis=1)

    rows: list[int] = []
    cols: list[int] = []
    for u, vs in objs["graph"].items():
        for v in vs:
            rows.append(u)
            cols.append(v)
    adj = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)
    )
    adj = ((adj + adj.T) > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj, features, labels
