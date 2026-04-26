"""Dataset loaders for Lambda Lens experiments.

All loaders return (adjacency CSR sparse, features dense float32 or None, labels int or None).
Cache_dir defaults to data/processed; downloads are idempotent and atomic via tmp+rename.
"""
from __future__ import annotations

import gzip
import os
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


def _atomic_download(url: str, dst: Path) -> None:
    """Download url to dst atomically via tmp+rename. Idempotent."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with ur.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, dst)


def _atomic_save_npy(arr: np.ndarray, dst: Path) -> None:
    """Save arr to dst (.npy) atomically. Bypasses np.save's suffix munging by using a file handle."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, dst)


def _atomic_save_npz(adj: sp.spmatrix, dst: Path) -> None:
    """Save sparse matrix atomically. scipy.sparse.save_npz auto-appends .npz too."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(tmp, "wb") as f:
        sp.save_npz(f, adj)
    os.replace(tmp, dst)


def load_planetoid(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Return (adjacency CSR, dense features float32, integer labels) for cora/citeseer/pubmed.

    Reconstructs node-id-aligned features and labels per the planetoid convention
    (test rows get placed at their original node IDs; isolated citeseer test nodes
    remain as zero-feature/label-0 placeholders).
    """
    cache = Path(cache_dir) / name
    objs: dict = {}
    for f in PLANETOID_FILES:
        url = f"{PLANETOID_BASE}/ind.{name}.{f}"
        path = cache / f"ind.{name}.{f}"
        _atomic_download(url, path)
        if f == "test.index":
            objs[f] = np.array([int(line) for line in path.read_text().split()])
        else:
            with path.open("rb") as fp:
                objs[f] = pickle.load(fp, encoding="latin1")

    test_idx_reorder = objs["test.index"]
    allx, tx = objs["allx"], objs["tx"]
    ally, ty = objs["ally"], objs["ty"]
    n_allx = allx.shape[0]
    n_feat = allx.shape[1]
    n_class = ally.shape[1]

    if name == "citeseer":
        n = int(test_idx_reorder.max()) + 1
    else:
        n = n_allx + tx.shape[0]

    features = np.zeros((n, n_feat), dtype=np.float32)
    features[:n_allx] = allx.toarray()
    tx_dense = tx.toarray() if sp.issparse(tx) else tx
    features[test_idx_reorder] = tx_dense.astype(np.float32)

    labels_oh = np.zeros((n, n_class), dtype=ally.dtype)
    labels_oh[:n_allx] = ally
    labels_oh[test_idx_reorder] = ty
    labels = labels_oh.argmax(axis=1)

    rows: list[int] = []
    cols: list[int] = []
    for u, vs in objs["graph"].items():
        for v in vs:
            if u < n and v < n:
                rows.append(u)
                cols.append(v)
    adj = sp.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)
    )
    adj = ((adj + adj.T) > 0).astype(np.float32)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj, features, labels


MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _parse_mnist_images(path: Path) -> np.ndarray:
    import struct

    with gzip.open(path, "rb") as f:
        _magic, n, h, w = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h * w)


def _parse_mnist_labels(path: Path) -> np.ndarray:
    import struct

    with gzip.open(path, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_knn(
    k: int = 15, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build a k-NN graph over MNIST-784 (raw IDX from PyTorch S3 mirror)."""
    from sklearn.neighbors import kneighbors_graph

    cache = Path(cache_dir) / "mnist_knn"
    cache.mkdir(parents=True, exist_ok=True)
    cache_npz = cache / f"mnist_knn_k{k}.npz"
    cache_feat = cache / "mnist_features.npy"
    cache_lbl = cache / "mnist_labels.npy"

    if not (cache_feat.exists() and cache_lbl.exists()):
        for fname in MNIST_FILES:
            _atomic_download(f"{MNIST_BASE}/{fname}", cache / fname)
        x_tr = _parse_mnist_images(cache / "train-images-idx3-ubyte.gz")
        x_te = _parse_mnist_images(cache / "t10k-images-idx3-ubyte.gz")
        y_tr = _parse_mnist_labels(cache / "train-labels-idx1-ubyte.gz")
        y_te = _parse_mnist_labels(cache / "t10k-labels-idx1-ubyte.gz")
        features = np.vstack([x_tr, x_te]).astype(np.float32)
        labels = np.concatenate([y_tr, y_te]).astype(int)
        _atomic_save_npy(features, cache_feat)
        _atomic_save_npy(labels, cache_lbl)
    else:
        features = np.load(cache_feat)
        labels = np.load(cache_lbl)

    if cache_npz.exists():
        adj = sp.load_npz(cache_npz)
    else:
        adj = kneighbors_graph(features, n_neighbors=k, mode="connectivity", n_jobs=-1)
        adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        _atomic_save_npz(adj, cache_npz)

    return adj, features, labels


def load_snap_edgelist(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray | None, None]:
    """Load a SNAP undirected edge list. No labels. Features=None for n>5000."""
    if name not in SNAP_URLS:
        raise ValueError(f"unknown SNAP dataset: {name}")
    cache = Path(cache_dir) / name
    cache.mkdir(parents=True, exist_ok=True)
    raw = cache / f"{name}.txt.gz"
    _atomic_download(SNAP_URLS[name], raw)

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
    features = adj.toarray().astype(np.float32) if n <= 5000 else None
    return adj, features, None


def load_dataset(
    name: str, cache_dir: str | Path = "data/processed"
) -> tuple[sp.csr_matrix, np.ndarray | None, np.ndarray | None]:
    """Dispatch to the right loader. None features means 'skip ZADU on this dataset'."""
    if name in ("cora", "citeseer", "pubmed"):
        return load_planetoid(name, cache_dir)
    if name == "mnist_knn":
        return load_mnist_knn(15, cache_dir)
    if name in ("ca_astroph", "wiki_vote"):
        return load_snap_edgelist(name, cache_dir)
    raise ValueError(f"unknown dataset: {name}")
