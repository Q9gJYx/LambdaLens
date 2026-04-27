"""P1-7: synthetic regime control.

BA (Barabasi-Albert, heterogeneous, scale-free) vs WS (Watts-Strogatz, regular,
small-world). Both n=2000. Run 4 lambda values * 3 seeds * 2 graphs at PCA-init.
Render 2-row x 4-panel `synthetic_regime_control.pdf` colored by node-degree
percentile (viridis). Report CV(d) per graph and pairwise Procrustes mean over
the lambda grid per (graph, seed).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp

LAMBDAS = (1.0, 5.0, 20.0, 80.0)
SEEDS = (42, 43, 44)
GRAPHS = ("BA", "WS")
N = 2000


def _build_graph(kind: str, seed: int = 42) -> sp.csr_matrix:
    if kind == "BA":
        G = nx.barabasi_albert_graph(N, 3, seed=seed)
    elif kind == "WS":
        G = nx.watts_strogatz_graph(N, 10, 0.1, seed=seed)
    else:
        raise ValueError(kind)
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)


def _worker(kind: str, lam: float, seed: int, out_root: str) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    from lens.init import pca_init
    from pysgtsnepi import sgtsnepi
    adj = _build_graph(kind, seed=42)
    Y0 = pca_init(adj, d=2, scale=1e-4, random_state=seed)
    t0 = time.perf_counter()
    Y = sgtsnepi(adj, d=2, lambda_=lam, random_state=seed, Y0=Y0,
                 unweighted_to_weighted=True)
    rt = time.perf_counter() - t0
    out_dir = Path(out_root) / "embeddings_synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{kind}_lam{lam}_seed{seed}.npy", Y)
    return {"kind": kind, "lambda": lam, "seed": seed, "runtime_s": rt}


def _procrustes(Y1: np.ndarray, Y2: np.ndarray) -> float:
    Y1c = Y1 - Y1.mean(axis=0, keepdims=True)
    Y2c = Y2 - Y2.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Y2c.T @ Y1c, full_matrices=False)
    R = U @ Vt
    Y2r = Y2c @ R
    diff = np.linalg.norm(Y1c - Y2r) / np.sqrt(Y1c.shape[0])
    bbox = Y1c.max(axis=0) - Y1c.min(axis=0)
    diag = float(np.sqrt((bbox ** 2).sum()))
    return diff / diag if diag > 0 else 0.0


def _render(out_root: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.0))
    for row, kind in enumerate(GRAPHS):
        adj = _build_graph(kind)
        deg = np.asarray(adj.sum(axis=1)).ravel()
        pct = np.argsort(np.argsort(deg)) / (len(deg) - 1)
        for col, lam in enumerate(LAMBDAS):
            ax = axes[row, col]
            Y = np.load(out_root / "embeddings_synth" / f"{kind}_lam{lam}_seed42.npy")
            ax.scatter(Y[:, 0], Y[:, 1], s=1.5, c=pct, cmap="viridis", alpha=0.75, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            ax.text(0.5, 1.02, f"$\\lambda{{=}}{int(lam)}$", transform=ax.transAxes,
                    fontsize=10, ha="center", va="bottom")
    fig.text(0.005, 0.74, f"BA  CV={_cv(_build_graph('BA')):.2f}", rotation=90, va="center", fontsize=9)
    fig.text(0.005, 0.30, f"WS  CV={_cv(_build_graph('WS')):.2f}", rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.92, bottom=0.02, wspace=0.06, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cv(adj: sp.csr_matrix) -> float:
    d = np.asarray(adj.sum(axis=1)).ravel()
    return float(d.std() / d.mean()) if d.mean() > 0 else float("nan")


def _summary(out_root: Path) -> dict:
    summary: dict = {"per_graph": {}}
    for kind in GRAPHS:
        adj = _build_graph(kind)
        cv = _cv(adj)
        per_seed_pp: dict[str, float] = {}
        for seed in SEEDS:
            Ys = [np.load(out_root / "embeddings_synth" / f"{kind}_lam{lam}_seed{seed}.npy") for lam in LAMBDAS]
            pp = []
            for i in range(len(Ys)):
                for j in range(i + 1, len(Ys)):
                    pp.append(_procrustes(Ys[i], Ys[j]))
            per_seed_pp[f"seed{seed}"] = float(np.mean(pp))
        summary["per_graph"][kind] = {
            "n": int(adj.shape[0]),
            "cv_d": cv,
            "pairwise_procrustes_mean_per_seed": per_seed_pp,
            "pairwise_procrustes_grand_mean": float(np.mean(list(per_seed_pp.values()))),
        }
    return summary


def main() -> int:
    out_root = Path("output")
    cells = [(g, l, s) for g in GRAPHS for l in LAMBDAS for s in SEEDS]
    print(f"[p1-7] running {len(cells)} cells", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_worker, g, l, s, str(out_root)): (g, l, s) for g, l, s in cells}
        for f in as_completed(futs):
            g, l, s = futs[f]
            try:
                r = f.result()
                print(f"[p1-7] {g} lam={l} seed={s} {r['runtime_s']:.1f}s", flush=True)
            except Exception as e:
                print(f"[p1-7] FAIL {g} lam={l} seed={s} {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    print(f"[p1-7] all cells done in {time.perf_counter() - t0:.1f}s", flush=True)

    _render(out_root, out_root / "figures" / "synthetic_regime_control")
    print("[p1-7] wrote synthetic_regime_control.pdf", flush=True)
    s = _summary(out_root)
    out_path = out_root / "tables" / "synthetic_regime_summary.json"
    with open(out_path, "w") as f:
        json.dump(s, f, indent=2)
    print(f"[p1-7] wrote {out_path}", flush=True)
    for k, v in s["per_graph"].items():
        print(f"  {k}: CV(d)={v['cv_d']:.3f} pairwise_proc_mean={v['pairwise_procrustes_grand_mean']:.4f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
