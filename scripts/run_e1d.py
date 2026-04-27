"""E1d expanded: PBMC at original-paper lambda range, with seed-noise + PCA-init.

Runs (lambda, seed, init) cells through the extended `run_one_cell`, with
`unweighted_to_weighted=False` (PBMC is already a stochastic kNN matrix).
Resumable via per-cell parquets under `output/tables/cells/`. After cells
finish: merges into `output/tables/{dataset}_e1d_full.parquet`, renders 6-panel
portraits per (init, seed=42), and writes pairwise Procrustes + seed-noise +
init-noise summary to `output/tables/{dataset}_e1d_procrustes.csv`.

Default: PBMC, lambda in {1, 5, 10, 20, 50, 80}, seeds {42, 43, 44}, inits
{random, pca}. Random-init runs all 3 seeds; PCA-init runs seed=42 only
(PCA-init is deterministic given the SVD random_state, so the seed is just
for any internal pysgtsnepi noise).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASETS = ("pbmc",)
DEFAULT_LAMBDAS = (1.0, 5.0, 10.0, 20.0, 50.0, 80.0)
DEFAULT_RANDOM_SEEDS = (42, 43, 44)
DEFAULT_PCA_SEEDS = (42,)


def _cell_filename(dataset: str, lam: float, seed: int, init: str, uw: bool) -> str:
    suf = []
    if init != "random":
        suf.append(f"init={init}")
    if not uw:
        suf.append("uw=False")
    suffix = ("_" + "_".join(suf)) if suf else ""
    return f"{dataset}_lam{lam}_seed{seed}{suffix}"


def _existing(out_root: Path, dataset: str) -> set[tuple[float, int, str, bool]]:
    cells_dir = out_root / "tables" / "cells"
    if not cells_dir.exists():
        return set()
    found: set[tuple[float, int, str, bool]] = set()
    for p in cells_dir.glob(f"{dataset}_*.parquet"):
        try:
            df = pd.read_parquet(p)
            for r in df.itertuples():
                init = str(getattr(r, "init", "random"))
                uw = bool(getattr(r, "unweighted_to_weighted", True))
                found.add((float(r.lambda_), int(r.seed), init, uw))
        except Exception:
            continue
    return found


def _worker(
    dataset: str,
    lambda_: float,
    seed: int,
    init: str,
    unweighted_to_weighted: bool,
    out_root: str,
    zadu_subsample_n: int,
) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell

    adj, features, labels = load_dataset(dataset)
    return run_one_cell(
        adj,
        features,
        labels,
        lambda_,
        seed,
        dataset,
        out_root,
        zadu_subsample_n,
        init=init,
        unweighted_to_weighted=unweighted_to_weighted,
    )


def _merge(out_root: Path, dataset: str) -> pd.DataFrame:
    cells_dir = out_root / "tables" / "cells"
    files = sorted(cells_dir.glob(f"{dataset}_*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in files]
    combined = pd.concat(dfs, ignore_index=True)
    if "init" not in combined.columns:
        combined["init"] = "random"
    if "unweighted_to_weighted" not in combined.columns:
        combined["unweighted_to_weighted"] = True
    combined = combined.drop_duplicates(
        subset=["dataset", "lambda_", "seed", "init", "unweighted_to_weighted"],
        keep="last",
    )
    combined = combined.sort_values(
        ["init", "unweighted_to_weighted", "lambda_", "seed"]
    ).reset_index(drop=True)
    out = out_root / "tables" / f"{dataset}_e1d_full.parquet"
    combined.to_parquet(out, index=False)
    return combined


def _procrustes(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Procrustes distance after centering + optimal rotation, normalized by Y1's bbox diagonal."""
    Y1c = Y1 - Y1.mean(axis=0, keepdims=True)
    Y2c = Y2 - Y2.mean(axis=0, keepdims=True)
    U, _, Vt = np.linalg.svd(Y2c.T @ Y1c, full_matrices=False)
    R = U @ Vt
    Y2r = Y2c @ R
    diff = np.linalg.norm(Y1c - Y2r) / np.sqrt(Y1c.shape[0])
    bbox = Y1c.max(axis=0) - Y1c.min(axis=0)
    diag = float(np.sqrt((bbox ** 2).sum()))
    return diff / diag if diag > 0 else 0.0


def _load_emb(out_root: Path, dataset: str, lam: float, seed: int, init: str, uw: bool) -> np.ndarray:
    name = _cell_filename(dataset, lam, seed, init, uw) + ".npy"
    return np.load(out_root / "embeddings" / name)


def _summarize(out_root: Path, dataset: str, lambdas: list[float], seeds: list[int], uw: bool) -> dict:
    """Pairwise Procrustes at seed=42 random init; seed-noise at lambda=10; init-noise pca-vs-random per lambda."""
    pair_random_seed42 = {}
    Y_lam = {l: _load_emb(out_root, dataset, l, 42, "random", uw) for l in lambdas}
    pair_distances = []
    for i, la in enumerate(lambdas):
        for lb in lambdas[i + 1 :]:
            d = _procrustes(Y_lam[la], Y_lam[lb])
            pair_random_seed42[f"{la}-{lb}"] = d
            pair_distances.append(d)

    seed_noise_lam = 10.0 if 10.0 in lambdas else lambdas[len(lambdas) // 2]
    Y_seed = [_load_emb(out_root, dataset, seed_noise_lam, s, "random", uw) for s in seeds]
    seed_noise_pairs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            seed_noise_pairs.append(_procrustes(Y_seed[i], Y_seed[j]))
    seed_noise = float(np.mean(seed_noise_pairs)) if seed_noise_pairs else float("nan")

    init_noise = {}
    for la in lambdas:
        try:
            Yr = _load_emb(out_root, dataset, la, 42, "random", uw)
            Yp = _load_emb(out_root, dataset, la, 42, "pca", uw)
            init_noise[str(la)] = _procrustes(Yr, Yp)
        except FileNotFoundError:
            init_noise[str(la)] = float("nan")

    mean_pair = float(np.mean(pair_distances)) if pair_distances else float("nan")
    max_pair = float(np.max(pair_distances)) if pair_distances else float("nan")
    ratio = mean_pair / seed_noise if seed_noise and not np.isnan(seed_noise) else float("nan")
    if max_pair > 0.15:
        verdict = "STRONG"
    elif max_pair >= 0.05:
        verdict = "MARGINAL"
    else:
        verdict = "WEAK"
    return {
        "dataset": dataset,
        "lambdas": lambdas,
        "seeds": seeds,
        "pair_random_seed42": pair_random_seed42,
        "mean_pairwise_procrustes": mean_pair,
        "max_pairwise_procrustes": max_pair,
        "seed_noise_lambda": seed_noise_lam,
        "seed_noise": seed_noise,
        "lambda_to_seed_ratio": ratio,
        "init_noise_pca_vs_random_at_seed42": init_noise,
        "verdict_max_gate": verdict,
    }


def _render_portrait(
    out_root: Path,
    dataset: str,
    lambdas: list[float],
    seed: int,
    init: str,
    uw: bool,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, lam in zip(axes, lambdas):
        try:
            Y = _load_emb(out_root, dataset, lam, seed, init, uw)
        except FileNotFoundError:
            ax.set_axis_off()
            ax.set_title(f"λ={lam} (missing)", fontsize=8)
            continue
        ax.scatter(Y[:, 0], Y[:, 1], s=1.5, c="#1f77b4", alpha=0.55, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"λ={lam}", fontsize=9)
    fig.suptitle(
        f"{dataset} — seed={seed} — init={init}" + ("" if uw else " — uw=False"),
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    p.add_argument("--random-seeds", nargs="+", type=int, default=list(DEFAULT_RANDOM_SEEDS))
    p.add_argument("--pca-seeds", nargs="+", type=int, default=list(DEFAULT_PCA_SEEDS))
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--out-root", default="output")
    p.add_argument("--zadu-subsample-n", type=int, default=5000)
    p.add_argument("--unweighted-to-weighted", action="store_true",
                   help="Default off: PBMC is already stochastic.")
    p.add_argument("--skip-run", action="store_true",
                   help="Skip the run loop; only merge + summarize + render.")
    p.add_argument("--state-json", default="output/meta/run_e1d_state.json")
    args = p.parse_args()

    out_root = Path(args.out_root)
    state_path = Path(args.state_json)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    uw = bool(args.unweighted_to_weighted)

    cells: list[tuple[str, float, int, str, bool]] = []
    for ds in args.datasets:
        done = _existing(out_root, ds)
        for lam in args.lambdas:
            for seed in args.random_seeds:
                key = (lam, seed, "random", uw)
                if key not in done:
                    cells.append((ds, lam, seed, "random", uw))
            for seed in args.pca_seeds:
                key = (lam, seed, "pca", uw)
                if key not in done:
                    cells.append((ds, lam, seed, "pca", uw))

    if not args.skip_run and cells:
        total = len(cells)
        print(f"[e1d] {total} cells to run on {args.max_workers} workers (uw={uw})", flush=True)
        t0 = time.perf_counter()
        completed = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(_worker, ds, lam, seed, init, _uw, args.out_root, args.zadu_subsample_n): (ds, lam, seed, init, _uw)
                for ds, lam, seed, init, _uw in cells
            }
            for fut in as_completed(futs):
                ds, lam, seed, init, _uw = futs[fut]
                try:
                    row = fut.result()
                    completed += 1
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / completed * (total - completed)
                    print(
                        f"[e1d] [{completed}/{total}] {ds} lam={lam} seed={seed} init={init} "
                        f"runtime={row['runtime_s']:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                        flush=True,
                    )
                    state_path.write_text(json.dumps({
                        "completed": completed, "total": total,
                        "elapsed_s": elapsed, "eta_s": eta,
                        "last": {"dataset": ds, "lambda_": lam, "seed": seed, "init": init},
                    }))
                except Exception as e:
                    print(f"[e1d] FAILED {ds} lam={lam} seed={seed} init={init}: {type(e).__name__}: {e}",
                          flush=True, file=sys.stderr)
    elif not cells:
        print("[e1d] nothing to run", flush=True)

    for ds in args.datasets:
        df = _merge(out_root, ds)
        if df.empty:
            print(f"[e1d] {ds}: no cells, skipping summary", flush=True)
            continue
        print(f"[e1d] {ds}: merged {len(df)} rows", flush=True)
        summary = _summarize(out_root, ds, list(args.lambdas), list(args.random_seeds), uw)
        with (out_root / "meta" / f"{ds}_e1d_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        rows = []
        for k, v in summary["pair_random_seed42"].items():
            la, lb = k.split("-")
            rows.append({"kind": "lambda_pair", "a": la, "b": lb, "procrustes_normalized": v})
        rows.append({"kind": "seed_noise", "a": f"lam={summary['seed_noise_lambda']}", "b": "seeds 42-44 mean", "procrustes_normalized": summary["seed_noise"]})
        for k, v in summary["init_noise_pca_vs_random_at_seed42"].items():
            rows.append({"kind": "init_noise_pca_vs_random", "a": f"lam={k} pca", "b": f"lam={k} random", "procrustes_normalized": v})
        pd.DataFrame(rows).to_csv(out_root / "tables" / f"{ds}_e1d_procrustes.csv", index=False)
        print(f"[e1d] {ds}: verdict={summary['verdict_max_gate']} max={summary['max_pairwise_procrustes']:.3f} "
              f"mean={summary['mean_pairwise_procrustes']:.3f} seed-noise={summary['seed_noise']:.3f}",
              flush=True)
        _render_portrait(out_root, ds, list(args.lambdas), 42, "random", uw,
                         out_root / "figures" / f"portrait_e1d_{ds}_seed42")
        _render_portrait(out_root, ds, list(args.lambdas), 42, "pca", uw,
                         out_root / "figures" / f"portrait_e1d_{ds}_pcainit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
