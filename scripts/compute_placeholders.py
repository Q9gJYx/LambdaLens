"""[Pa] [Pb] [Pc] [Pd] paper-side placeholders.

[Pa] auto-lambda runtime on PBMC: time the 4-fit grid lambda in {1, 5, 20, 50}
     at PCA-init seed 42 (sequential, total wall). Per round-3 spec these
     four fits constitute the auto-lambda call.
[Pb] degree-moment fit lambda = c0 + c1 * CV(d), least-squares on the 3
     labeled datasets (Cora=20, Citeseer=10, MNIST-kNN=20).
[Pc] predicted lambda for PBMC.
[Pd] predicted lambda for ca-AstroPh.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lens.data import load_dataset
from lens.init import pca_init

PA_LAMBDAS = (1.0, 5.0, 20.0, 50.0)
PA_SEED = 42


def _time_pa() -> dict:
    from pysgtsnepi import sgtsnepi

    adj, _, _ = load_dataset("pbmc")
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=PA_SEED)
    times: dict[str, float] = {}
    t_total0 = time.perf_counter()
    for lam in PA_LAMBDAS:
        t0 = time.perf_counter()
        _ = sgtsnepi(adj, d=2, lambda_=lam, random_state=PA_SEED, Y0=Y0,
                     unweighted_to_weighted=False)
        times[f"lambda_{lam}"] = time.perf_counter() - t0
    total = time.perf_counter() - t_total0
    return {
        "pbmc_4fit_pca_init_seed42_seconds": float(total),
        "per_lambda_s": times,
        "lambda_grid": list(PA_LAMBDAS),
        "init": "pca",
        "seed": PA_SEED,
    }


def _moment_fit(cv_summary: dict[str, dict]) -> dict:
    pts = [
        ("cora", cv_summary["cora"]["cv_d"], 20.0),
        ("citeseer", cv_summary["citeseer"]["cv_d"], 10.0),
        ("mnist_knn", cv_summary["mnist_knn"]["cv_d"], 20.0),
    ]
    cv = np.array([p[1] for p in pts])
    lam = np.array([p[2] for p in pts])
    X = np.column_stack([np.ones_like(cv), cv])
    coef, *_ = np.linalg.lstsq(X, lam, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])
    pred = c0 + c1 * cv
    ss_res = float(((lam - pred) ** 2).sum())
    ss_tot = float(((lam - lam.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def _predict(d: str) -> float:
        v = c0 + c1 * cv_summary[d]["cv_d"]
        return float(np.clip(v, 1.0, 80.0))

    return {
        "c0": c0,
        "c1": c1,
        "r_squared": r2,
        "predicted_pbmc": _predict("pbmc"),
        "predicted_ca_astroph": _predict("ca_astroph"),
        "predicted_pubmed": _predict("pubmed") if "pubmed" in cv_summary else None,
        "inputs": {p[0]: {"cv_d": p[1], "lambda_grid_best": p[2]} for p in pts},
        "method": "ordinary_least_squares_3pt",
        "note": ("3-point fit on grid-best lambda for labeled datasets per round-3 spec; "
                 "Citeseer uses gridsearch lambda=10, not auto-lambda=5."),
    }


def main() -> int:
    cv_path = Path("output/tables/cv_d_summary.json")
    if not cv_path.exists():
        raise SystemExit("[placeholders] need cv_d_summary.json — run scripts/cv_d_summary.py first")
    with open(cv_path) as f:
        cv_summary = json.load(f)

    pa = _time_pa()
    Path("output/meta").mkdir(parents=True, exist_ok=True)
    with open("output/meta/pa_runtime.json", "w") as f:
        json.dump(pa, f, indent=2)
    print(f"[Pa] PBMC 4-fit auto-lambda runtime = {pa['pbmc_4fit_pca_init_seed42_seconds']:.2f} s "
          f"(per-lambda: {pa['per_lambda_s']})", flush=True)

    fit = _moment_fit(cv_summary)
    with open("output/tables/moment_fit.json", "w") as f:
        json.dump(fit, f, indent=2)
    print(f"[Pb] lambda = {fit['c0']:.3f} + {fit['c1']:.3f} * CV(d), R^2 = {fit['r_squared']:.3f}", flush=True)
    print(f"[Pc] predicted PBMC lambda = {fit['predicted_pbmc']:.2f}", flush=True)
    print(f"[Pd] predicted ca_astroph lambda = {fit['predicted_ca_astroph']:.2f}", flush=True)
    if fit.get("predicted_pubmed") is not None:
        print(f"[bonus] predicted PubMed lambda = {fit['predicted_pubmed']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
