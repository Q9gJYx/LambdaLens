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


def run_one_cell(
    adj: sp.csr_matrix,
    features: np.ndarray,
    labels: np.ndarray | None,
    lambda_: float,
    seed: int,
    dataset: str,
    out_root: str | Path = "output",
) -> dict[str, Any]:
    """Run SG-t-SNE-Pi at one (lambda, seed); persist embedding + parquet row."""
    out_root = Path(out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "embeddings").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=lambda_, random_state=seed)
    runtime_s = time.perf_counter() - t0

    spec: list[dict[str, Any]] = [
        {"id": "trustworthiness_continuity", "params": {"k": 15}}
    ]
    if labels is not None:
        spec.append(
            {"id": "label_trustworthiness_and_continuity", "params": {"cvm": "dsc"}}
        )
    raw = ZADU(spec, features).measure(Y, label=labels)

    metrics: dict[str, float] = {}
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
