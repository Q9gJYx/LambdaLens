"""Dataset loaders for Lambda Lens experiments.

All loaders return (adjacency CSR sparse, features dense float32, labels int or None).
Cache_dir defaults to data/processed; downloads are idempotent.
"""
from __future__ import annotations

import gzip
import io
import pickle
import urllib.request as ur
from pathlib import Path

import numpy as np
import scipy.sparse as sp

PLANETOID_BASE = "https://github.com/kimiyoung/planetoid/raw/master/data"
PLANETOID_FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")

SNAP_URLS = {
    "ca_astroph": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "wiki_vote": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with ur.urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())


def load_planetoid(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (adjacency CSR, dense features, integer labels) for cora/citeseer/pubmed."""
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

    # Citeseer has isolated test nodes — pad tx/ty to include them as zero rows.
    if name == "citeseer":
        full = np.arange(int(test_idx.min()), int(test_idx.max()) + 1)
        tx_pad = sp.lil_matrix((len(full), allx.shape[1]), dtype=tx.dtype)
        tx_pad[test_idx_sorted - full.min(), :] = tx
        tx = tx_pad
        ty_pad = np.zeros((len(full), ally.shape[1]), dtype=ty.dtype)
        ty_pad[test_idx_sorted - full.min(), :] = ty
        ty = ty_pad
        test_idx_sorted = full

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


def load_mnist_knn(
    k: int = 15, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build a k-NN graph over MNIST-784. Cached as .npz."""
    from sklearn.datasets import fetch_openml
    from sklearn.neighbors import kneighbors_graph

    cache = Path(cache_dir) / "mnist_knn"
    cache.mkdir(parents=True, exist_ok=True)
    cache_npz = cache / f"mnist_knn_k{k}.npz"
    cache_feat = cache / "mnist_features.npy"
    cache_lbl = cache / "mnist_labels.npy"

    if cache_feat.exists() and cache_lbl.exists():
        features = np.load(cache_feat)
        labels = np.load(cache_lbl)
    else:
        ds = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)
        features = ds.data.astype(np.float32)
        labels = ds.target.astype(int)
        np.save(cache_feat, features)
        np.save(cache_lbl, labels)

    if cache_npz.exists():
        adj = sp.load_npz(cache_npz)
    else:
        adj = kneighbors_graph(features, n_neighbors=k, mode="connectivity", n_jobs=-1)
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        sp.save_npz(cache_npz, adj)

    return adj, features, labels


def load_snap_edgelist(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, None]:
    """Load a SNAP undirected edge list. No labels; HD 'features' = adjacency rows."""
    if name not in SNAP_URLS:
        raise ValueError(f"unknown SNAP dataset: {name}")
    cache = Path(cache_dir) / name
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / f"{name}.txt.gz"
    _download(SNAP_URLS[name], raw)

    edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    with gzip.open(raw, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            iu, iv = int(u), int(v)
            edges.append((iu, iv))
            nodes.add(iu)
            nodes.add(iv)

    node_list = sorted(nodes)
    remap = {n: i for i, n in enumerate(node_list)}
    rows = np.array([remap[u] for u, _ in edges], dtype=np.int64)
    cols = np.array([remap[v] for _, v in edges], dtype=np.int64)
    n = len(node_list)
    adj = sp.csr_matrix(
        (np.ones(len(edges), np.float32), (rows, cols)), shape=(n, n)
    )
    adj = ((adj + adj.T) > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    # No node features; pass adjacency-row vectors as HD coords (sparse->dense limited use).
    # ZADU needs dense HD; we will subsample anyway for large graphs.
    features = adj.toarray().astype(np.float32) if n <= 5000 else None
    return adj, features, None


def load_dataset(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray | None, np.ndarray | None]:
    """Dispatch to the right loader. None features means 'use adjacency-row distance'."""
    if name in ("cora", "citeseer", "pubmed"):
        return load_planetoid(name, cache_dir)
    if name == "mnist_knn":
        return load_mnist_knn(15, cache_dir)
    if name in ("ca_astroph", "wiki_vote"):
        return load_snap_edgelist(name, cache_dir)
    raise ValueError(f"unknown dataset: {name}")
