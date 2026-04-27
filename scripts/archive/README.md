# scripts/archive/

One-off recovery / finalization utilities used during the round-2 and round-3 push.
Their results are already merged into `output/tables/*_comparison_agg.parquet`
(R3 N=5 multi-seed) and the canonical pipeline does not depend on them. Kept
for reproducibility and audit.

- `_resurrect_phase6_node2vec.py` — copy Phase-6 single-seed node2vec_umap
  results for `ca_astroph` into `cells_baselines/` after the R3 T6 PHATE
  subsample produced a disconnected subgraph. T/C recomputed with
  `lens.metrics.compute_metrics`. Result lives in
  `output/tables/cells_baselines/ca_astroph_node2vec_umap_seed42.parquet`.

- `_round3_finalize.py` — paper-ready summary printer; reads the merged
  `*_comparison_agg.parquet` and prints the §5 table for paper-side. Result
  is the printed summary plus `output/meta/round3_final_summary.json`.

- `_run_missing_pca_cells.py` — one-shot fill of λ ∈ {1, 5, 80} PCA-init
  seed=42 cells for Cora and MNIST-kNN (needed for `fig:lens` insets). Result
  lives in `output/embeddings/<ds>_lam<λ>_seed42_init=pca.npy` and the
  matching parquet rows.

- `_run_pubmed_node2vec_fast.py` — one-shot PubMed `node2vec_umap` single-seed
  with `workers=4` (the canonical R3 runner forces `workers=1` to avoid
  thread oversubscription inside the ProcessPool; for a single isolated cell
  we can crank workers up). Result lives in
  `output/embeddings_baselines/pubmed_node2vec_umap_seed42.npy`.

Re-running these scripts requires CWD = repo root and the same input parquets
that existed at round-3 time; they are not idempotent across changes to the
canonical runners.
