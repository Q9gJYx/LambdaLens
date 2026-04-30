# Lambda Lens — experiments repo

Companion experiments and artifacts for the IEEE VIS 2026 short paper
**Lambda Lens** on the λ parameter of SG-t-SNE-Π. The paper itself
lives in a separate Overleaf repo; this tree holds the `lens` analysis
package, the runner scripts, and the per-experiment artifact trees.

Working branch: `paper-vis2026`. `main` stays clean.

## Quick start

```bash
uv sync                       # base env (auto-resolves .python-version = 3.11)
uv sync --extra baselines     # adds umap-learn, opentsne, phate, node2vec, ogb, hdbscan
uv run pytest -q              # smoke test (lens importable)
bash init.sh                  # session bootstrap: env + state + git status
```

`pysgtsnepi` is pinned to commit `qqgjyx/sgtsnepi@b1131f8` via
`[tool.uv.sources]` in `pyproject.toml`; the PyPI 0.3.0 wheel omits the
`unweighted_to_weighted` Jaccard preprocessing without which
`lambda_rescaling` is a no-op on unweighted symmetrized graphs. Do not
edit that pin without re-reading the comment.

## Layout

```
.
├── src/lens/                # analysis package (data loaders, run_one_cell, metrics, pca_init)
├── scripts/                 # canonical entry points (one per experiment); see Reproduce below
│   └── archive/             # one-off recovery utilities from rounds 2 and 3
├── data/
│   ├── raw/, interim/, external/   # gitignored, populated by loaders
│   └── processed/                  # PBMC graph + spectral-Agg labels (committed); planetoid loaders cache here
├── output/
│   ├── embeddings/          # E1 / E1d cells, gitignored
│   ├── embeddings_baselines/ # E3 cells, gitignored
│   ├── embeddings_synth/    # P1-7 synthetic cells, gitignored
│   ├── tables/              # parquet/json artifacts, gitignored
│   ├── figures/             # PDF/PNG, gitignored
│   └── meta/                # logs (gitignored) + state snapshots + cached PCA-init Y0
├── tests/test_smoke.py
├── PAPER_VIS2026.md         # paper-side coordination surface (status + requests)
├── CLAUDE.md                # agent operating rules for this repo
├── state.json               # machine-readable experiment state
├── init.sh                  # session bootstrap
└── pyproject.toml, uv.lock
```

Embedding and table artifacts are gitignored on purpose: the (dataset, λ,
seed, method) cells are reproducible from `scripts/` and the loaders, and
they total ~80 MB which we don't want in git.

## Reproduce

All scripts run from the repo root via `uv run python scripts/<name>.py`.
Each one is resume-safe via per-cell parquets under
`output/tables/cells*/`; re-running skips finished cells.

### E1: λ-grid effect

| script | datasets | grid | seeds | output |
|---|---|---|---|---|
| `run_e1.py` | one cell | one λ | one seed | `output/tables/cells/<ds>_lambda_grid_lam<λ>_seed<s>.parquet` + embedding |
| `run_e1_local.py` | 4 default | {0.5, 1, 2, 5, 10, 20} | {42, 43, 44} | `output/tables/<ds>_lambda_grid.parquet` |
| `run_e1d.py` | PBMC | {1, 5, 10, 20, 50, 80} | {42, 43, 44} × {random, pca} | `output/tables/pbmc_e1d_full.parquet`, `pbmc_e1d_procrustes.csv`, `output/figures/portrait_e1d_pbmc_*.pdf` |

### E2: auto-λ heuristic

| script | what it does |
|---|---|
| `run_e2_auto_lambda.py` | argmax over {1, 5, 20, 50} of Label-T&C (or T&C). Fills missing cells. Writes `output/tables/auto_lambda_summary.parquet`. |
| `run_pubmed_e1_e2.py` | PubMed E1 grid + refresh of the auto-λ summary across all 6 datasets. |

### E3: baseline comparison

| script | what it does |
|---|---|
| `run_e3_baselines.py` | N=5 seeds per (method, dataset) for pysgtsnepi / openTSNE / UMAP / PHATE / node2vec+UMAP, with PCA-init throughout, matched n_iter budgets, graph-only routing. Per-cell parquets under `output/tables/cells_baselines/`. |
| `merge_e3_results.py` | merges cell parquets into `output/tables/<ds>_comparison.parquet` (per-seed rows) and `<ds>_comparison_agg.parquet` (mean ± std + best-of-3). |

### Round-3 deliverables

| script | output |
|---|---|
| `run_p0_3a_pysgtsnepi.py` | pysgtsnepi multi-seed at auto-λ + PCA-init across the 5 R3 datasets |
| `derive_pbmc_labels.py` | `data/processed/pbmc/labels.npy` (Agglomerative-Ward k=7 on top-15 normalized-Laplacian eigenvectors of the input kNN graph; λ-independent) |
| `run_p1_7_synth.py` | BA vs WS regime control: `output/figures/synthetic_regime_control.pdf` + `synthetic_regime_summary.json` |
| `cv_d_summary.py` | `output/tables/cv_d_summary.json` (per-dataset CV(d), feeds [Pb] moment fit) |
| `compute_placeholders.py` | `[Pa] [Pb] [Pc] [Pd]` in `output/tables/moment_fit.json` |
| `render_p0_figures.py` | `output/figures/teaser_pbmc_hero.pdf`, `lens_idiom_regimes.pdf` |

### Archived (one-shot)

`scripts/archive/` holds four recovery utilities used during R2/R3.
Their outputs are already merged into the comparison parquets; the
canonical pipeline does not depend on them. See
`scripts/archive/README.md` for the index.

## Datasets

| dataset | n | source | resolved by |
|---|---|---|---|
| Cora | 2 708 | Planetoid (PyG) | `lens.data.load_cora` (network fetch + cache to `data/processed/cora/`) |
| Citeseer | 3 327 | Planetoid (PyG) | `lens.data.load_citeseer` |
| PubMed | 19 717 | Planetoid (PyG) | `lens.data.load_pubmed` |
| MNIST-kNN | 70 000 | torchvision S3 mirror, k=15 cosine kNN | `lens.data.load_mnist_knn` |
| ca-AstroPh | 18 772 | SNAP edge list | `lens.data.load_ca_astroph` |
| PBMC-8k | 8 381 | `pbmc-graph.tar.gz` from `fcdimitr/sgtsnepi`, committed under `data/processed/pbmc/` | `lens.data.load_pbmc` |
| ogbn-arxiv | 169 343 | OGB | `lens.data.load_ogbn_arxiv` (R3 loader, pipeline pending) |

PBMC labels are derived via Agglomerative-Ward k=7 on the top-15
non-trivial eigenvectors of the symmetric normalized Laplacian of the
input kNN graph (Ng-Jordan-Weiss row-normalization). This is
λ-independent — clusters reflect graph topology rather than any
particular embedding's geometry, so Label-T&C scored against these
labels does not favor any specific λ by construction. Zheng-2017 PBMC-8k
labels for the n=8381 fcdimitr split are not publicly mirrored; see
`derive_pbmc_labels.py` and `data/processed/pbmc/labels_provenance.json`
for full provenance. All other loaders fetch on demand and cache
idempotently.

## Project state

- **`PAPER_VIS2026.md`** is the authoritative status surface; the
  paper-side agent reads it for round-by-round handoffs. New
  experiment-side updates go under a fresh "Paper-side ← experiment-side,
  round N" sub-heading at the bottom.
- **`state.json`** is machine state: compute hosts, dataset / λ / seed
  grids, per-experiment progress flags, session log.
- **`CLAUDE.md`** is the agent operating manual for this repo (three-repo
  split rules, branch convention, do-not list).

## Compute

Workloads are CPU-bound (`pysgtsnepi`, PHATE, UMAP, openTSNE) and run on
either local M3 Pro (8 workers) or `zjl` (64 cores). The runners use
`concurrent.futures.ProcessPoolExecutor`; set `OMP_NUM_THREADS=1` per
process to avoid thread oversubscription, and pass `workers=1` to
`Node2Vec` inside the ProcessPool. Pull artifacts back from `zjl` after
each tier closes:

```bash
rsync -avz zjl:/home/zhoujunlin/WorkSpace/wh/SGtSNE-Pi/output/ ./output/
```

## License

MIT. See `LICENSE`.
