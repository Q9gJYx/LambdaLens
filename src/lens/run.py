"""Single-cell runner: one (dataset, lambda, seed, init, uw) tuple end-to-end.

Persists outputs in a race-free manner:
- embeddings via atomic tmp+rename (`_atomic_save_npy` from data.py)
- per-cell parquet rows under tables/cells/ (no read-modify-write race)
- a separate merge step (in run_e1_local.py) rolls cells into per-dataset parquets

Filenames suffix only when the kwarg differs from default, so existing E1
artifacts keep their schema and resumability is preserved.

Skipped-metric cells (ZADU off; e.g. unlabeled large SNAP graphs) emit explicit
NaN sentinels so all cells share schema.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pysgtsnepi import sgtsnepi
from zadu import ZADU

from lens.data import _atomic_save_npy
from lens.init import pca_init

METRIC_COLUMNS = (
    "trustworthiness",
    "continuity",
    "label_trustworthiness",
    "label_continuity",
)


def _zadu_subsample(
    features: np.ndarray | None,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    n = embedding.shape[0]
    if features is None or n <= max_n:
        return features, embedding, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    sub_labels = labels[idx] if labels is not None else None
    return features[idx], embedding[idx], sub_labels


def _suffix(init: str, unweighted_to_weighted: bool) -> str:
    parts = []
    if init != "random":
        parts.append(f"init={init}")
    if not unweighted_to_weighted:
        parts.append("uw=False")
    return ("_" + "_".join(parts)) if parts else ""


def run_one_cell(
    adj: sp.csr_matrix,
    features: np.ndarray | None,
    labels: np.ndarray | None,
    lambda_: float,
    seed: int,
    dataset: str,
    out_root: str | Path = "output",
    zadu_subsample_n: int = 5000,
    init: Literal["random", "pca"] = "random",
    Y0_scale: float = 1e-4,
    unweighted_to_weighted: bool = True,
) -> dict[str, Any]:
    out_root = Path(out_root)
    (out_root / "tables" / "cells").mkdir(parents=True, exist_ok=True)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)

    Y0 = pca_init(adj, d=2, scale=Y0_scale, random_state=seed) if init == "pca" else None

    t0 = time.perf_counter()
    Y = sgtsnepi(
        adj,
        d=2,
        lambda_=lambda_,
        random_state=seed,
        Y0=Y0,
        unweighted_to_weighted=unweighted_to_weighted,
    )
    runtime_s = time.perf_counter() - t0

    metrics: dict[str, float] = {c: float("nan") for c in METRIC_COLUMNS}
    if features is not None:
        sub_feat, sub_Y, sub_lbl = _zadu_subsample(
            features, Y, labels, zadu_subsample_n, seed
        )
        spec: list[dict[str, Any]] = [
            {"id": "trustworthiness_continuity", "params": {"k": 15}}
        ]
        if sub_lbl is not None:
            spec.append(
                {
                    "id": "label_trustworthiness_and_continuity",
                    "params": {"cvm": "dsc"},
                }
            )
        raw = ZADU(spec, sub_feat).measure(sub_Y, label=sub_lbl)
        for entry in raw:
            for k, v in entry.items():
                if k in METRIC_COLUMNS:
                    metrics[k] = float(v)

    suf = _suffix(init, unweighted_to_weighted)
    emb_path = out_root / "embeddings" / f"{dataset}_lam{lambda_}_seed{seed}{suf}.npy"
    if emb_path.exists():
        emb_path.unlink()
    _atomic_save_npy(Y, emb_path)

    row = {
        "dataset": dataset,
        "lambda_": float(lambda_),
        "seed": int(seed),
        "init": init,
        "unweighted_to_weighted": bool(unweighted_to_weighted),
        "runtime_s": runtime_s,
        **metrics,
    }
    cell_path = (
        out_root / "tables" / "cells" / f"{dataset}_lam{lambda_}_seed{seed}{suf}.parquet"
    )
    pd.DataFrame([row]).to_parquet(cell_path, index=False)

    return row
