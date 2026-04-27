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


def _moment_fit(cv_summary: dict[str, dict],
                auto_lambda_path: Path | None = None) -> dict:
    """OLS fit on labeled-dataset grid-best lambda values.

    R4-B expands from 3 to 5 labeled datasets by adding PubMed and PBMC.
    Grid-best lambda read from auto_lambda_summary.parquet when available
    and expanded grid has been run; falls back to R3 hardcoded values.
    """
    # Base 3-point inputs (R3, always available)
    base_pts = [
        ("cora", cv_summary["cora"]["cv_d"], 20.0),
        ("citeseer", cv_summary["citeseer"]["cv_d"], 10.0),
        ("mnist_knn", cv_summary["mnist_knn"]["cv_d"], 20.0),
    ]
    extra_pts: list[tuple[str, float, float]] = []
    if auto_lambda_path is not None and auto_lambda_path.exists():
        al_df = pd.read_parquet(auto_lambda_path) if str(auto_lambda_path).endswith(".parquet") else None
        if al_df is not None:
            for ds in ("pubmed", "pbmc"):
                if ds not in cv_summary:
                    continue
                row = al_df[al_df["dataset"] == ds]
                if row.empty:
                    continue
                v = row["auto_lambda"].iloc[0]
                if v and not (isinstance(v, float) and np.isnan(v)):
                    extra_pts.append((ds, cv_summary[ds]["cv_d"], float(v)))

    pts = base_pts + extra_pts
    cv = np.array([p[1] for p in pts])
    lam = np.array([p[2] for p in pts])
    X = np.column_stack([np.ones_like(cv), cv])
    coef, *_ = np.linalg.lstsq(X, lam, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])
    pred = c0 + c1 * cv
    ss_res = float(((lam - pred) ** 2).sum())
    ss_tot = float(((lam - lam.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    def _predict(d: str) -> float | None:
        if d not in cv_summary:
            return None
        v = c0 + c1 * cv_summary[d]["cv_d"]
        return float(np.clip(v, 1.0, 80.0))

    n_pts = len(pts)
    return {
        "c0": c0,
        "c1": c1,
        "r_squared": r2,
        "n_fit_points": n_pts,
        "predicted_pbmc": _predict("pbmc"),
        "predicted_ca_astroph": _predict("ca_astroph"),
        "predicted_pubmed": _predict("pubmed"),
        "inputs": {p[0]: {"cv_d": p[1], "lambda_grid_best": p[2]} for p in pts},
        "method": f"ordinary_least_squares_{n_pts}pt",
        "note": (f"{n_pts}-point fit; Citeseer uses gridsearch lambda=10 (not auto=5); "
                 "PubMed+PBMC added if auto_lambda_summary available (R4-B)."),
    }


def main() -> int:
    import pandas as pd

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

    al_path = Path("output/tables/auto_lambda_summary.parquet")
    fit = _moment_fit(cv_summary, al_path if al_path.exists() else None)
    with open("output/tables/moment_fit.json", "w") as f:
        json.dump(fit, f, indent=2)
    print(f"[Pb] lambda = {fit['c0']:.3f} + {fit['c1']:.3f} * CV(d), "
          f"R^2 = {fit['r_squared']:.3f} ({fit['n_fit_points']}-pt fit)", flush=True)
    if fit.get("predicted_pbmc") is not None:
        print(f"[Pc] predicted PBMC lambda = {fit['predicted_pbmc']:.2f}", flush=True)
    if fit.get("predicted_ca_astroph") is not None:
        print(f"[Pd] predicted ca_astroph lambda = {fit['predicted_ca_astroph']:.2f}", flush=True)
    if fit.get("predicted_pubmed") is not None:
        print(f"[bonus] predicted PubMed lambda = {fit['predicted_pubmed']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
