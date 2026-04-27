"""R4-F: subtractive ablation (DriftNeRF-style).

Variants:
  full:                        pysgtsnepi(auto-lambda, PCA-init) -- from comparison_agg
  pca_init_off:                pysgtsnepi(auto-lambda, random init) -- new compute
  auto_lambda_off_fixed20:     pysgtsnepi(lambda=20, PCA-init) -- from lambda_grid at lam=20
  degree_rescaling_off_lambda1: pysgtsnepi(lambda=1, PCA-init) -- from lambda_grid at lam=1

Output: output/tables/ablation.parquet + ablation_table.tex
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LABELED_DATASETS = ["cora", "citeseer", "pubmed", "mnist_knn", "pbmc"]
SEEDS = [42, 43, 44, 45, 46]


def _worker_random_init(ds: str, auto_lam: float, seed: int, out_root: str) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from lens.data import load_dataset
    from lens.run import run_one_cell
    adj, features, labels = load_dataset(ds)
    uw = ds != "pbmc"
    return run_one_cell(adj, features, labels, auto_lam, seed, ds, out_root,
                        init="random", unweighted_to_weighted=uw)


def _extract_from_grid(ds: str, lam: float, seeds: list[int],
                       cells_dir: Path, init: str, uw: bool) -> list[dict]:
    suf = ""
    if init != "random":
        suf += f"_init={init}"
    if not uw:
        suf += "_uw=False"
    rows = []
    for seed in seeds:
        p = cells_dir / f"{ds}_lam{lam}_seed{seed}{suf}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if not df.empty:
                rows.append(df.iloc[0].to_dict())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=LABELED_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--out-root", default="output")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    cells_dir = out_root / "tables" / "cells"
    tables = out_root / "tables"

    # load auto-lambda mapping
    auto_lam_path = tables / "auto_lambda_summary.parquet"
    auto_lam_map: dict[str, float] = {}
    if auto_lam_path.exists():
        al_df = pd.read_parquet(auto_lam_path)
        for _, row in al_df.iterrows():
            v = row.get("auto_lambda")
            if v and not (isinstance(v, float) and np.isnan(v)):
                auto_lam_map[row["dataset"]] = float(v)

    all_rows: list[dict] = []

    # F2: auto_lambda_off (lambda=20, PCA-init) -- extract from existing cells
    # F3: degree_rescaling_off (lambda=1, PCA-init) -- extract from existing cells
    for ds in args.datasets:
        uw = ds != "pbmc"
        for variant, lam in [("auto_lambda_off_fixed20", 20.0),
                              ("degree_rescaling_off_lambda1", 1.0)]:
            extracted = _extract_from_grid(ds, lam, args.seeds, cells_dir,
                                           init="pca", uw=uw)
            for r in extracted:
                all_rows.append({
                    "dataset": ds, "variant": variant, "seed": r.get("seed"),
                    "label_T": r.get("label_trustworthiness", float("nan")),
                    "label_C": r.get("label_continuity", float("nan")),
                    "runtime_s": r.get("runtime_s", float("nan")),
                })
        # "full" variant: from comparison_agg
        agg_path = tables / f"{ds}_comparison_agg.parquet"
        if agg_path.exists():
            agg_df = pd.read_parquet(agg_path)
            pysg_row = agg_df[agg_df["method"] == "pysgtsnepi"]
            if not pysg_row.empty:
                r = pysg_row.iloc[0]
                all_rows.append({
                    "dataset": ds, "variant": "full", "seed": None,
                    "label_T": r.get("label_trustworthiness_mean", float("nan")),
                    "label_C": r.get("label_continuity_mean", float("nan")),
                    "runtime_s": r.get("runtime_s_mean", float("nan")),
                })

    # F1: pca_init_off -- new compute (random init, auto-lambda)
    cells_f1 = []
    for ds in args.datasets:
        al = auto_lam_map.get(ds, 20.0)
        for seed in args.seeds:
            # check if random-init cell already exists in cells/
            suf = "_uw=False" if ds == "pbmc" else ""
            p = cells_dir / f"{ds}_lam{al}_seed{seed}{suf}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                row = df[(df["init"] == "random")] if "init" in df.columns else df
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    all_rows.append({
                        "dataset": ds, "variant": "pca_init_off", "seed": int(seed),
                        "label_T": r.get("label_trustworthiness", float("nan")),
                        "label_C": r.get("label_continuity", float("nan")),
                        "runtime_s": r.get("runtime_s", float("nan")),
                    })
                    continue
            cells_f1.append((ds, al, seed))

    if cells_f1:
        print(f"[F1] {len(cells_f1)} random-init cells to run", flush=True)
        t0 = time.perf_counter()
        done = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(_worker_random_init, ds, al, seed, args.out_root): (ds, seed)
                for ds, al, seed in cells_f1
            }
            for fut in as_completed(futs):
                ds, seed = futs[fut]
                try:
                    r = fut.result()
                    done += 1
                    all_rows.append({
                        "dataset": ds, "variant": "pca_init_off", "seed": int(seed),
                        "label_T": r.get("label_trustworthiness", float("nan")),
                        "label_C": r.get("label_continuity", float("nan")),
                        "runtime_s": r.get("runtime_s", float("nan")),
                    })
                    print(f"[F1] [{done}/{len(cells_f1)}] {ds} seed={seed} "
                          f"LT={r.get('label_trustworthiness', float('nan')):.3f} "
                          f"elapsed={(time.perf_counter()-t0)/60:.1f}m", flush=True)
                except Exception as e:
                    print(f"[F1] FAIL {ds} seed={seed}: {e}", file=sys.stderr, flush=True)

    df = pd.DataFrame(all_rows)
    out_path = tables / "ablation.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[F] wrote {out_path} ({len(df)} rows)", flush=True)

    # aggregate mean/std per (dataset, variant)
    agg = df.groupby(["dataset", "variant"]).agg(
        label_T_mean=("label_T", "mean"),
        label_T_std=("label_T", "std"),
        label_C_mean=("label_C", "mean"),
        runtime_s_mean=("runtime_s", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()

    # emit tex
    tex_lines = ["% ablation table rows (auto-generated)"]
    variants_order = ["full", "pca_init_off", "auto_lambda_off_fixed20",
                      "degree_rescaling_off_lambda1"]
    variant_names = {
        "full": r"\ours (full)",
        "pca_init_off": r"\quad -- PCA init",
        "auto_lambda_off_fixed20": r"\quad -- auto-$\lambda$ ($\lambda{=}20$)",
        "degree_rescaling_off_lambda1": r"\quad -- degree rescaling ($\lambda{=}1$)",
    }
    for ds in args.datasets:
        tex_lines.append(f"\n% {ds}")
        tex_lines.append(r"\midrule")
        tex_lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{ds}}}}} \\\\")
        sub = agg[agg["dataset"] == ds]
        for v in variants_order:
            row = sub[sub["variant"] == v]
            if row.empty:
                continue
            r = row.iloc[0]
            lt = f"{r['label_T_mean']:.3f}"
            std = f"$\\pm${r['label_T_std']:.3f}" if not np.isnan(r.get('label_T_std', float('nan'))) else ""
            tex_lines.append(f"  {variant_names[v]} & {lt}{std} & {r['label_C_mean']:.3f} & {r['runtime_s_mean']:.1f} \\\\")

    tex_path = tables / "ablation_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"[F] wrote {tex_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
