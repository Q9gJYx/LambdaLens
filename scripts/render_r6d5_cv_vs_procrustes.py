"""R6-D5: CV(d) vs mean pairwise Procrustes scatter.

For each main dataset, computes mean pairwise Procrustes across the 16-pt
PCA-init seed=42 lambda grid (already on disk). Adds BA + WS synthetic
controls from synthetic_regime_summary.json. Plots one point per dataset
with horizontal seed-noise band (when multi-seed data available, namely
seeds {42-46} at lambdas {1,5,20,80} from R4-A insets).

Vertical bands shade the three regimes at CV(d) <= 0.3 (regular),
0.3 < CV(d) <= 1.0 (marginal), CV(d) > 1.0 (heterogeneous).
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes

LAMBDAS_16PT = (0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 100, 150)
DATASETS_UW = (("cora", True), ("citeseer", True), ("pubmed", True),
               ("mnist_knn", True), ("ca_astroph", True))
DATASETS_UWFALSE = (("pbmc", False),)
SEED_NOISE_LAMBDAS = (1.0, 5.0, 20.0, 80.0)  # R4-A multi-seed coverage


def _emb_path(ds: str, lam: float, seed: int, uw: bool) -> Path:
    suf = "" if uw else "_uw=False"
    return Path(f"output/embeddings/{ds}_lam{lam}_seed{seed}_init=pca{suf}.npy")


def _normalized_procrustes(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Orthogonal Procrustes (rotation+reflection, no scale), normalized by RMS
    radius of Y1. Returns RMSE(Y1, R*Y2) / sqrt(mean(||Y1 - mean(Y1)||^2)).

    This is the standard 'shape distance' used in the paper's R3 numbers.
    Values typically in [0, ~0.5] for distinct embeddings of the same data.
    """
    if Y1.shape != Y2.shape:
        return float("nan")
    Y1 = Y1.astype(np.float64); Y2 = Y2.astype(np.float64)
    Y1c = Y1 - Y1.mean(axis=0)
    Y2c = Y2 - Y2.mean(axis=0)
    # Optimal orthogonal R minimizing ||Y1c - Y2c R||_F: SVD of Y2c^T Y1c
    U, _, Vt = np.linalg.svd(Y2c.T @ Y1c, full_matrices=False)
    R = U @ Vt
    Y2_rot = Y2c @ R
    rmse = float(np.sqrt(((Y1c - Y2_rot) ** 2).sum() / Y1c.shape[0]))
    rms_radius = float(np.sqrt((Y1c ** 2).sum() / Y1c.shape[0]))
    return rmse / rms_radius if rms_radius > 0 else float("nan")


def _mean_pairwise_proc(ds: str, uw: bool) -> float:
    """Mean pairwise normalized Procrustes across all 16 lambda values at seed=42."""
    embs = []
    for lam in LAMBDAS_16PT:
        p = _emb_path(ds, lam, 42, uw)
        if p.exists():
            embs.append(np.load(p))
    if len(embs) < 2:
        return float("nan")
    n_pairs = 0
    total = 0.0
    for a, b in combinations(embs, 2):
        v = _normalized_procrustes(a, b)
        if not np.isnan(v):
            total += v; n_pairs += 1
    return total / n_pairs if n_pairs > 0 else float("nan")


def _seed_noise_floor(ds: str, uw: bool) -> float:
    """Mean pairwise Procrustes between seeds at fixed lambda, averaged over
    the 4 lambdas in SEED_NOISE_LAMBDAS where R4-A has seeds {42-46}."""
    floors = []
    for lam in SEED_NOISE_LAMBDAS:
        seed_embs = []
        for s in (42, 43, 44, 45, 46):
            p = _emb_path(ds, lam, s, uw)
            if p.exists():
                seed_embs.append(np.load(p))
        if len(seed_embs) < 2:
            continue
        pairs = [_normalized_procrustes(a, b)
                 for a, b in combinations(seed_embs, 2)]
        valid = [v for v in pairs if not np.isnan(v)]
        if valid:
            floors.append(float(np.mean(valid)))
    return float(np.mean(floors)) if floors else float("nan")


def main() -> int:
    cv = json.load(open("output/tables/cv_d_summary.json"))
    synth_path = Path("output/tables/synthetic_regime_summary.json")
    synth = json.load(open(synth_path)) if synth_path.exists() else {}

    points = []  # (label, cv, proc, noise, color, marker)
    print("[D5] computing mean pairwise Procrustes (this takes ~5-15 min)...", flush=True)
    for ds, uw in (*DATASETS_UW, *DATASETS_UWFALSE):
        if ds not in cv:
            continue
        cv_d = cv[ds]["cv_d"]
        proc = _mean_pairwise_proc(ds, uw)
        noise = _seed_noise_floor(ds, uw)
        print(f"  {ds}: CV(d)={cv_d:.3f} mean_proc={proc:.4f} seed_noise={noise:.4f}",
              flush=True)
        points.append((ds, cv_d, proc, noise, "C0", "o"))

    # Synthetic BA / WS
    for tag, color, marker in [("ba", "C2", "^"), ("ws", "C3", "v")]:
        if tag in synth:
            d = synth[tag]
            cv_d = float(d.get("cv_d", float("nan")))
            proc = float(d.get("pairwise_proc_mean", float("nan")))
            label = f"{tag.upper()}-synth"
            print(f"  {label}: CV(d)={cv_d:.3f} mean_proc={proc:.4f}", flush=True)
            points.append((label, cv_d, proc, float("nan"), color, marker))

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    # regime bands
    ax.axvspan(0.05, 0.3, alpha=0.07, color="C0", zorder=0)
    ax.axvspan(0.3, 1.0, alpha=0.07, color="C1", zorder=0)
    ax.axvspan(1.0, 3.0, alpha=0.07, color="C3", zorder=0)
    ax.text(0.17, ax.transAxes.transform((0, 1))[1] * 0.95 if False else 0.45,
            "regular", fontsize=7, color="gray", ha="center", va="top",
            transform=ax.get_xaxis_transform())
    ax.text(0.55, 0.45, "marginal", fontsize=7, color="gray",
            ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(1.6, 0.45, "heterogeneous", fontsize=7, color="gray",
            ha="center", va="top", transform=ax.get_xaxis_transform())

    for label, cv_d, proc, noise, color, marker in points:
        if np.isnan(proc) or np.isnan(cv_d):
            continue
        if not np.isnan(noise):
            ax.errorbar(cv_d, proc, xerr=None, yerr=noise, fmt="none",
                        ecolor=color, alpha=0.4, capsize=2, linewidth=0.8,
                        zorder=2)
        ax.scatter(cv_d, proc, c=color, marker=marker, s=40, alpha=0.9,
                   edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(label, (cv_d, proc), xytext=(5, 3),
                    textcoords="offset points", fontsize=7, color="black")

    ax.set_xlabel(r"CV($d$)", fontsize=9)
    ax.set_ylabel(r"mean pairwise Procrustes", fontsize=9)
    ax.set_xscale("log")
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.05, 3.0)

    fig.tight_layout()
    out = Path("output/figures/cv_vs_procrustes")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[D5] wrote {out.with_suffix('.pdf')}", flush=True)

    # Save numeric summary so paper-side can verify
    summary = {p[0]: {"cv_d": p[1], "mean_proc": p[2], "seed_noise": p[3]}
               for p in points}
    Path("output/tables/cv_vs_procrustes_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("[D5] wrote output/tables/cv_vs_procrustes_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
