"""Emit doc/r9_ablation_cells.md from output/tables/ablation_v1.parquet.

Format per R9 spec:
  - Top-3 per column ranked with \\first / \\second / \\third
  - Std formatted via \\s{...}
  - Full-system row uses \\rowcolor{gray!15} instead of rank colors
  - PBMC -- Jaccard cell rendered as `--` (N/A)

Output: doc/r9_ablation_cells.md  (paper-side reads; not a paper-repo write)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROW_ORDER = ["full", "no_pca", "no_jaccard", "lam1", "lam20"]
ROW_LABELS = {
    "full": r"\ours (full)",
    "no_pca": r"\quad $-$ PCA init",
    "no_jaccard": r"\quad $-$ Jaccard",
    "lam1": r"\quad $\lambda{=}1$ (no rescale)",
    "lam20": r"\quad $\lambda{=}20$ (heuristic ceiling)",
}
DATASETS = ["cora", "pbmc"]
DATASET_HEADERS = {"cora": "Cora", "pbmc": "PBMC-8k"}


def _fmt_cell(val: float, std: float, rank_macro: str | None, n_seeds: int) -> str:
    if np.isnan(val):
        return "$--$"
    if np.isnan(std):
        std_part = ""
    else:
        std_part = r"\s{" + f"{std:.3f}" + "}"
    body = f"${val:.3f}$" + std_part
    if n_seeds == 1:
        body += r"$^{\dagger}$"
    if rank_macro is None:
        return body
    return r"\\" + rank_macro + "{" + body + "}"  # placeholder, actual prefix is one backslash


def _rank_macro_factory():
    """Returns a function that takes a sorted desc list and yields rank macro names."""
    def rank_for(idx: int) -> str | None:
        if idx == 0:
            return "first"
        if idx == 1:
            return "second"
        if idx == 2:
            return "third"
        return None
    return rank_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="output/tables/ablation_v1.parquet")
    ap.add_argument("--out", default="doc/r9_ablation_cells.md")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    df = pd.read_parquet(parquet)

    # Aggregate per (dataset, variant)
    agg = (
        df.groupby(["dataset", "variant"])
        .agg(
            mean_T=("label_T", "mean"),
            std_T=("label_T", "std"),
            mean_C=("label_C", "mean"),
            std_C=("label_C", "std"),
            mean_rt=("runtime_s", "mean"),
            n=("seed", "count"),
        )
        .reset_index()
    )

    # Build per-(dataset, metric) ranking on label_T (R9 spec: top-3 per column).
    # Exclude `full` from rank (uses gray rowcolor instead). Rank by descending mean_T.
    rank_for = _rank_macro_factory()
    rank_map: dict[tuple[str, str], str | None] = {}
    for ds in DATASETS:
        sub = agg[(agg["dataset"] == ds) & (agg["variant"] != "full")].copy()
        sub = sub.sort_values("mean_T", ascending=False, kind="stable")
        ranked_variants = list(sub["variant"])
        for i, v in enumerate(ranked_variants):
            rank_map[(ds, v)] = rank_for(i)

    # Emit markdown reply file
    lines: list[str] = []
    lines.append("# R9 D4 — Ready-to-paste LaTeX cells")
    lines.append("")
    lines.append(
        "Source: `output/tables/ablation_v1.parquet` "
        f"(N={int(agg['n'].max())} seeds 42-46, post-R8.1 PBMC labels). "
        "Aggregated to `output/tables/ablation_summary.csv`. "
        "Rank macros (`\\first / \\second / \\third`) applied per column on Label-T&C, "
        "excluding the full-system row (which uses `\\rowcolor{gray!15}`). "
        "PBMC `-- Jaccard` is N/A (PBMC ships as stochastic kNN; no Jaccard step in pipeline)."
    )
    lines.append("")
    lines.append("```latex")
    lines.append("% R9 D4 — paste between \\toprule and \\bottomrule of the ablation tabular.")
    lines.append("% Required macros (in preamble or t03_ablation.tex):")
    lines.append("%   \\newcommand{\\first}[1]{\\textbf{#1}}")
    lines.append("%   \\newcommand{\\second}[1]{\\underline{#1}}")
    lines.append("%   \\newcommand{\\third}[1]{\\textit{#1}}")
    lines.append(
        "%   \\newcommand{\\s}[1]{{\\scriptsize\\,$\\pm$\\,#1}}"
    )
    lines.append("%   (and \\rowcolor support via \\usepackage{colortbl,xcolor})")
    lines.append("")
    lines.append("% header (one column per dataset, one Label-T value cell)")
    lines.append("% Variant & Cora L-T\\&C & PBMC L-T\\&C \\\\")
    lines.append("\\midrule")
    for variant in ROW_ORDER:
        cells: list[str] = []
        for ds in DATASETS:
            row = agg[(agg["dataset"] == ds) & (agg["variant"] == variant)]
            if row.empty:
                # PBMC -- Jaccard: render as --
                cells.append("$--$")
                continue
            r = row.iloc[0]
            mean_T = float(r["mean_T"])
            std_T = float(r["std_T"]) if not np.isnan(r["std_T"]) else float("nan")
            n = int(r["n"])
            if variant == "full":
                # gray row: no rank macro
                std_part = "" if np.isnan(std_T) else r"\s{" + f"{std_T:.3f}" + "}"
                body = f"${mean_T:.3f}$" + std_part
                if n == 1:
                    body += r"$^{\dagger}$"
                cells.append(body)
            else:
                rk = rank_map.get((ds, variant))
                std_part = "" if np.isnan(std_T) else r"\s{" + f"{std_T:.3f}" + "}"
                body = f"${mean_T:.3f}$" + std_part
                if n == 1:
                    body += r"$^{\dagger}$"
                if rk is not None:
                    body = "\\" + rk + "{" + body + "}"
                cells.append(body)
        prefix = (
            r"\rowcolor{gray!15} " + ROW_LABELS[variant]
            if variant == "full"
            else ROW_LABELS[variant]
        )
        lines.append(prefix + " & " + " & ".join(cells) + r" \\")
    lines.append("```")
    lines.append("")

    # Append a raw summary table for reference
    lines.append("## Aggregated numbers (for reconciliation)")
    lines.append("")
    lines.append("| Dataset | Variant | Label-T mean | Label-T std | Label-C mean | runtime mean (s) | n seeds |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        for variant in ROW_ORDER:
            row = agg[(agg["dataset"] == ds) & (agg["variant"] == variant)]
            if row.empty:
                lines.append(f"| {DATASET_HEADERS[ds]} | {variant} | -- | -- | -- | -- | 0 |")
                continue
            r = row.iloc[0]
            std_T = "--" if np.isnan(r["std_T"]) else f"{float(r['std_T']):.4f}"
            lines.append(
                f"| {DATASET_HEADERS[ds]} | {variant} | "
                f"{float(r['mean_T']):.4f} | {std_T} | "
                f"{float(r['mean_C']):.4f} | {float(r['mean_rt']):.1f} | {int(r['n'])} |"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- For Cora, the auto-$\\lambda$ heuristic resolves to 20 (per R5-A1), so the "
        "`\\ours (full)` and `\\lambda{=}20` rows are produced by the same configuration "
        "and yield identical Label-T&C numbers (within float noise). Paper-side may "
        "footnote: \"On Cora, auto-$\\lambda$ selects $\\lambda{=}20$, so rows 1 and 5 are "
        "the same configuration.\""
    )
    lines.append(
        "- PBMC `-- Jaccard` is N/A: PBMC ships as a stochastic kNN graph and the "
        "`unweighted_to_weighted` Jaccard preprocessing step is not applied in the "
        "pipeline. Rendered as `$--$`."
    )
    lines.append(
        "- All cells use PCA-init (where applicable) with N=5 seeds (42-46). The N=5 "
        "PBMC numbers reflect the post-R8.1 spectral-Laplacian Agglomerative-Ward $k{=}7$ "
        "labels."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[R9-D4] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
