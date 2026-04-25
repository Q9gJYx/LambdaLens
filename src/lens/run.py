"""Single-cell runner: one (dataset, lambda, seed) tuple end-to-end."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pysgtsnepi import sgtsnepi
from zadu import ZADU


def _zadu_subsample(
    features: np.ndarray | None,
    embedding: np.ndarray,
    labels: np.ndarray | None,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    """Take a fixed random subset for ZADU when the graph is too big."""
    n = embedding.shape[0]
    if features is None or n <= max_n:
        return features, embedding, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    sub_labels = labels[idx] if labels is not None else None
    return features[idx], embedding[idx], sub_labels


def run_one_cell(
    adj: sp.csr_matrix,
    features: np.ndarray | None,
    labels: np.ndarray | None,
    lambda_: float,
    seed: int,
    dataset: str,
    out_root: str | Path = "output",
    zadu_subsample_n: int = 5000,
) -> dict[str, Any]:
    """Run SG-t-SNE-Pi at one (lambda, seed); persist embedding + parquet row."""
    out_root = Path(out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=lambda_, random_state=seed)
    runtime_s = time.perf_counter() - t0

    metrics: dict[str, float] = {}
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
            metrics.update({k: float(v) for k, v in entry.items()})

    np.save(
        out_root / "embeddings" / f"{dataset}_lam{lambda_}_seed{seed}.npy", Y
    )

    row = {
        "dataset": dataset,
        "lambda_": lambda_,
        "seed": seed,
        "runtime_s": runtime_s,
        **metrics,
    }
    parquet = out_root / "tables" / f"{dataset}_lambda_grid.parquet"
    new_df = pd.DataFrame([row])
    if parquet.exists():
        existing = pd.read_parquet(parquet)
        existing = existing[
            ~((existing["lambda_"] == lambda_) & (existing["seed"] == seed))
        ]
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_parquet(parquet, index=False)

    return row
