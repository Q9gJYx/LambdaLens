"""R5-A1/A2/A3: re-emit auto-lambda + moment fit from FIXED-build cells.

Paper-side discovered that the round-3 `auto_lambda_summary.parquet` was
calibrated on cells from the broken PyPI `pysgtsnepi v0.3.0` build
(unweighted_to_weighted Jaccard preprocessing missing -> lambda_rescaling
no-op on unweighted symmetrized graphs). The R4-B 16-pt grid cells used
the FIXED git rev `qqgjyx/sgtsnepi@b1131f8`. This script reads those
PCA-init seed=42 cells (the only seed in the 16-pt grid) and:

  R5-A1: argmax over Label-T&C harmonic mean -> auto-lambda per dataset.
  R5-A2: OLS fit lambda_moment = c0 + c1 * CV(d) on the LABELED corrected
         auto-lambda values (Cora, Citeseer, PubMed, MNIST-kNN, [PBMC if
         labels available]).
  R5-A3: predict lambda_moment for the unlabeled / held-out datasets
         (PBMC, ca_astroph) using the new (c0, c1).

Idempotent: writes to output/tables/auto_lambda_summary.parquet
                       output/tables/moment_fit.json (with provenance keys).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Auto-lambda probe set (R3 spec preserved; argmax over the 4 probe values)
PROBE_LAMBDAS = (1.0, 5.0, 20.0, 50.0)
# Full 16-pt grid (R4-B) for "gridsearch" comparison
GRIDSEARCH_LAMBDAS = (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
                      15.0, 20.0, 30.0, 50.0, 80.0, 100.0, 150.0)
SEED = 42


def _read_grid_cells(cells_dir: Path, ds: str, uw: bool,
                     init: str = "pca", recompute: bool = False) -> dict[float, dict]:
    """Read PCA-init seed=42 cells for ds at all available lambdas. Returns {lam: row_dict}.

    For graph-only datasets (features=None), the cell parquet has NaN metrics
    because run_one_cell skips ZADU when features are missing. We recompute
    label_T&C / T&C from the saved embedding + adj + (HDBSCAN) labels via
    lens.metrics.compute_metrics in those cases.
    """
    suf = ""
    if init != "random":
        suf += f"_init={init}"
    if not uw:
        suf += "_uw=False"
    emb_dir = cells_dir.parent.parent / "embeddings"
    out: dict[float, dict] = {}
    for p in sorted(cells_dir.glob(f"{ds}_lam*_seed{SEED}{suf}.parquet")):
        try:
            df = pd.read_parquet(p)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row_init = str(row.get("init", "random"))
            row_uw = bool(row.get("unweighted_to_weighted", True))
            if row_init != init or row_uw != uw:
                continue
            # If recompute requested AND metrics are NaN, recompute graph-T&C
            lt = float(row.get("label_trustworthiness", float("nan")))
            t = float(row.get("trustworthiness", float("nan")))
            if recompute and np.isnan(lt) and np.isnan(t):
                lam = float(row["lambda_"])
                emb_path = emb_dir / f"{ds}_lam{lam}_seed{SEED}{suf}.npy"
                if emb_path.exists():
                    Y = np.load(str(emb_path))
                    from lens.data import load_dataset
                    from lens.metrics import compute_metrics
                    if not hasattr(_read_grid_cells, "_cached_ds"):
                        _read_grid_cells._cached_ds = {}
                    if ds not in _read_grid_cells._cached_ds:
                        _read_grid_cells._cached_ds[ds] = load_dataset(ds)
                    adj, features, labels = _read_grid_cells._cached_ds[ds]
                    m = compute_metrics(features=features, adj=adj, Y=Y,
                                        labels=labels, max_n=2000, seed=SEED)
                    row.update(m)
                    print(f"    {ds} lam={lam}: lt={m.get('label_trustworthiness', float('nan')):.3f} "
                          f"lc={m.get('label_continuity', float('nan')):.3f}", flush=True)
            out[float(row["lambda_"])] = row
        except Exception as e:
            print(f"  ! cell read fail {p.name}: {e}")
            continue
    return out


def _harmonic(lt: float, lc: float) -> float:
    if np.isnan(lt) or np.isnan(lc) or lt <= 0 or lc <= 0:
        return float("nan")
    return 2.0 * lt * lc / (lt + lc)


def _argmax_over(cells: dict[float, dict], lambdas: tuple,
                 metric: str) -> tuple[float | None, float]:
    """Return (best_lambda, best_value); metric in {'label_TC', 'TC'}."""
    best_lam, best_val = None, -1.0
    for lam in lambdas:
        if lam not in cells:
            continue
        r = cells[lam]
        if metric == "label_TC":
            lt = float(r.get("label_trustworthiness", float("nan")))
            lc = float(r.get("label_continuity", float("nan")))
        else:
            lt = float(r.get("trustworthiness", float("nan")))
            lc = float(r.get("continuity", float("nan")))
        v = _harmonic(lt, lc)
        if not np.isnan(v) and v > best_val:
            best_lam, best_val = lam, v
    return best_lam, best_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    cells_dir = out_root / "tables" / "cells"
    tables = out_root / "tables"

    DATASETS = {
        "cora":       {"uw": True,  "labeled": True,  "compute": False},
        "citeseer":   {"uw": True,  "labeled": True,  "compute": False},
        "mnist_knn":  {"uw": True,  "labeled": True,  "compute": False},
        "pubmed":     {"uw": True,  "labeled": True,  "compute": False},
        "pbmc":       {"uw": False, "labeled": True,  "compute": True},   # graph-only with HDBSCAN labels
        "ca_astroph": {"uw": True,  "labeled": False, "compute": False},  # no labels: auto-lambda undefined
        "ogbn_arxiv": {"uw": True,  "labeled": True,  "compute": False},
    }

    rows: list[dict] = []
    for ds, cfg in DATASETS.items():
        print(f"  reading {ds} cells...", flush=True)
        cells = _read_grid_cells(cells_dir, ds, cfg["uw"], init="pca",
                                 recompute=cfg.get("compute", False))
        if not cells:
            rows.append({
                "dataset": ds, "auto_lambda": float("nan"),
                "auto_metric": float("nan"),
                "gridsearch_lambda": float("nan"),
                "gridsearch_metric": float("nan"),
                "match": False, "metric_used": "no_pca_cells",
                "metric_def": "harmonic_mean",
                "n_pca_cells": 0,
                "build": "qqgjyx/sgtsnepi@b1131f8 (FIXED)",
            })
            print(f"  {ds}: NO pca-init seed=42 cells found (need to rsync from zjl)")
            continue

        metric = "label_TC" if cfg["labeled"] else "TC"
        auto_lam, auto_val = _argmax_over(cells, PROBE_LAMBDAS, metric)
        grid_lam, grid_val = _argmax_over(cells, GRIDSEARCH_LAMBDAS, metric)
        rows.append({
            "dataset": ds,
            "auto_lambda": auto_lam if auto_lam else float("nan"),
            "auto_metric": auto_val if auto_lam else float("nan"),
            "gridsearch_lambda": grid_lam if grid_lam else float("nan"),
            "gridsearch_metric": grid_val if grid_lam else float("nan"),
            "match": auto_lam == grid_lam,
            "metric_used": "label_T&C" if cfg["labeled"] else "T&C",
            "metric_def": "harmonic_mean",
            "n_pca_cells": len(cells),
            "build": "qqgjyx/sgtsnepi@b1131f8 (FIXED)",
        })
        print(f"  {ds}: auto-lam={auto_lam} (val={auto_val:.4f}), "
              f"gridsearch={grid_lam} (val={grid_val:.4f}), "
              f"n_cells={len(cells)}")

    df = pd.DataFrame(rows)
    out = tables / "auto_lambda_summary.parquet"
    df.to_parquet(out, index=False)
    print(f"\n[R5-A1] wrote {out}")
    print(df.to_string(index=False))

    # R5-A2: re-fit moment OLS on labeled corrected values
    cv_path = tables / "cv_d_summary.json"
    if not cv_path.exists():
        print("[R5-A2] cv_d_summary.json missing; skipping moment refit")
        return 0
    cv = json.load(open(cv_path))

    fit_pts = []
    for _, row in df.iterrows():
        ds = row["dataset"]
        if not DATASETS[ds]["labeled"]:
            continue
        if np.isnan(row["auto_lambda"]):
            continue
        if ds not in cv:
            continue
        fit_pts.append((ds, cv[ds]["cv_d"], float(row["auto_lambda"])))

    if len(fit_pts) < 2:
        print(f"[R5-A2] only {len(fit_pts)} fit points; need >=2")
        return 0

    cv_arr = np.array([p[1] for p in fit_pts])
    lam_arr = np.array([p[2] for p in fit_pts])
    X = np.column_stack([np.ones_like(cv_arr), cv_arr])
    coef, *_ = np.linalg.lstsq(X, lam_arr, rcond=None)
    c0, c1 = float(coef[0]), float(coef[1])
    pred = c0 + c1 * cv_arr
    ss_res = float(((lam_arr - pred) ** 2).sum())
    ss_tot = float(((lam_arr - lam_arr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # R5-A3: predict lambda_moment for held-out datasets
    def _predict(ds: str) -> float | None:
        if ds not in cv:
            return None
        v = c0 + c1 * cv[ds]["cv_d"]
        return float(np.clip(v, 1.0, 80.0))

    fit_out = {
        "c0": c0, "c1": c1, "r_squared": r2,
        "n_fit_points": len(fit_pts),
        "method": f"ordinary_least_squares_{len(fit_pts)}pt",
        "predicted_pbmc": _predict("pbmc"),
        "predicted_ca_astroph": _predict("ca_astroph"),
        "predicted_pubmed": _predict("pubmed"),
        "predicted_ogbn_arxiv": _predict("ogbn_arxiv"),
        "inputs": {p[0]: {"cv_d": p[1], "auto_lambda": p[2]} for p in fit_pts},
        "metric_def": "harmonic_mean",
        "build": "qqgjyx/sgtsnepi@b1131f8 (FIXED)",
        "provenance": "R5-A1/A2/A3 re-emission from PCA-init seed=42 cells; "
                      "supersedes R3 3-pt fit calibrated on broken-build cells.",
    }
    out_fit = tables / "moment_fit.json"
    with open(out_fit, "w") as f:
        json.dump(fit_out, f, indent=2)
    print(f"\n[R5-A2] lambda = {c0:.3f} + {c1:.3f} * CV(d), R^2 = {r2:.3f} "
          f"({len(fit_pts)}-pt fit)")
    for ds in ("pbmc", "ca_astroph", "pubmed", "ogbn_arxiv"):
        v = _predict(ds)
        if v is not None:
            print(f"[R5-A3] predicted {ds} lambda = {v:.2f}")
    print(f"[R5-A2] wrote {out_fit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
