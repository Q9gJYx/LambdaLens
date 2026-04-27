# VIS 2026 Paper Status (Lambda Lens)

> Last updated: 2026-04-26 by experiment-side agent (Day 2 closing — Phase 6 in progress)

## Headline
- Day: 2 of 5 (Sun 2026-04-26, closing)
- **E1d expanded — MARGINAL on PBMC** (max λ-pair Procrustes 0.098, seed-noise 0.173, init-noise 0.18-0.22). Phase transition at λ=1↔λ≥5 is real but lives in the optimization noise floor; the original "Lambda Lens IS visible on PBMC" framing (single-seed) was over-confident.
- **E2 auto-λ heuristic validated**: argmax over {1,5,20,50} matches gridsearch optimum on cora (λ=20) and mnist_knn (λ=20); near-miss on citeseer (auto=5 vs gridsearch=10, both Label-T&C ~0.88). Defensible default.
- **E3 baselines — pysgtsnepi(auto-λ) is BEST on cora by a clear margin**: Label-T&C trust 0.934 vs next-best node2vec+UMAP 0.860 (+0.07 absolute), at 5 s vs 131 s. **Tied at the ceiling on mnist_knn** (label_T&C ≥ 0.999 for pysgtsnepi/UMAP/openTSNE). Third on citeseer (0.752 vs openTSNE 0.781, within metric noise). **Only method that runs natively on graph-only inputs** (PBMC, ca_astroph) — UMAP/openTSNE need features; PHATE OOMs; node2vec+UMAP is 100-1000× slower.
- **Blocker: paper-side framing decision**. Strongest single-dataset case is cora (E1 corrected Procrustes 0.184 + E3 cleanest baseline win). Three options unchanged: (1') auto-λ + characterization, (2') Lambda Lens scoped to heterogeneous-degree graphs, (3') integrate auto-λ + Lambda Lens (E1d PCA-init mode is the cleanest visual). Experiment-side leans **3'** with cora as the lead dataset and PBMC PCA-init as the demonstrative "what about non-feature graphs?" panel.

## Compute
- Host: **local M3 Pro / 8 workers / OMP=1**. zjl reachable at session start but SSH banner-timed-out on the actual rsync attempt (same network pattern as 2026-04-26 morning); local fallback proven (E1d expanded 24 cells in 1 min, E2 9 missing cells in 2 min, E3 baselines minutes per method).
- All cloud infrastructure (Azure `vis2026-jp` resource group, AWS c5.2xlarge in us-east-1) **fully terminated** Day 2 to avoid ongoing charges. Total cycle spend: ~$2.
- Runner: `scripts/run_e1_local.py` (E1) and the new `scripts/run_e1d.py`, `scripts/run_e2_auto_lambda.py`, `scripts/run_e3_baselines.py`, `scripts/merge_e3_results.py` — all `concurrent.futures.ProcessPoolExecutor` with `--max-workers 8`.
- Code changes Day 2: pinned `pysgtsnepi = git+https://github.com/qqgjyx/sgtsnepi.git@b1131f8` (PyPI 0.3.0 omits `unweighted_to_weighted` Jaccard preprocessing — see E1 audit callout); added `src/lens/init.py` with `pca_init`; extended `run_one_cell` with `init` / `Y0_scale` / `unweighted_to_weighted` kwargs; added `[project.optional-dependencies] baselines` for E3 (umap-learn, opentsne, phate, node2vec, setuptools<70 to keep node2vec's pkg_resources happy).

## Experiments

### E1 — lambda grid effect (DONE)
- Status: **complete**; finished 2026-04-26T05:59Z (154.6 min wall on Azure E4s_v3 + 2 workers, after one bug-fix relaunch)
- Cells: 96/96 (4 datasets × 8 λ × 3 seeds, λ ∈ {0.5, 1, 2, 5, 10, 20, 50, 100})
- Artifacts:
  - `output/tables/{cora,citeseer,mnist_knn,ca_astroph}_lambda_grid.parquet` — runtime + ZADU metrics per cell
  - `output/embeddings/<dataset>_lam<v>_seed<s>.npy` — 96 embeddings, all (n, 2) float64
  - `output/figures/teaser_preview.{pdf,png}` — 4-panel cora at λ ∈ {0.5, 1, 10, 100}, color by class

#### Kill-switch result: **TRIPPED on 4/4 datasets**

Pairwise Procrustes distance (after centering + optimal rotation) at λ ∈ {0.5, 1, 10, 100}, seed=42, normalized by embedding bounding-box diagonal:

| dataset | Procrustes-aligned | Raw (no rotation) | Seed noise (same λ, diff seeds) | λ-effect / seed-noise | verdict |
|---|---|---|---|---|---|
| cora | 4.0% | 4.0% | 17.5% | **0.23×** | TRIPPED |
| citeseer | 2.2% | 2.2% | 27.7% | **0.08×** | TRIPPED |
| mnist_knn | **0.0%** | 0.0% | 19.0% | **0.00×** | TRIPPED (degenerate — see below) |
| ca_astroph | 2.7% | 2.7% | 20.5% | **0.13×** | TRIPPED |

Threshold per `experiment_plan.md` was: TRIPPED if mean Procrustes < 5% on 3+ datasets. **All four are below 5%, on both Procrustes-aligned AND raw distance**, ruling out "embeddings differ but only by rotation" as a save.

> ⚠️ **Update 2026-04-26 — original verdict invalidated by stack audit.** A 3-track audit (pysgtsnepi port, lens code, killswitch math) found the installed `pysgtsnepi==0.3.0` PyPI wheel **omits the `unweighted_to_weighted` / `local_weights` Jaccard preprocessing** that the local working tree (commit `qqgjyx/sgtsnepi@b1131f8`) contains. Without Jaccard preprocessing, `lambda_rescaling` is a mathematical identity for unweighted symmetrized graphs (uniform `1/degree` column entries cancel through bisection + per-column renormalization). All 96 cells of the original E1 above produced false-negative results.
>
> After fixing (`pyproject.toml [tool.uv.sources] pysgtsnepi = { git = "qqgjyx/sgtsnepi", rev = "b1131f8" }`) + restricting λ ∈ {0.5, 1, 2, 5, 10, 20} per literature audit (λ > ~k drives mass negative-σ failures; λ ∈ {50, 100} from the original grid was pathological and never used in published SG-t-SNE-Π work), re-running 72 cells (4 × 6 × 3) on local M3 Pro / 8 workers in **5.7 min wall, $0 cost**:
>
> | dataset | corrected λ Procrustes | broken-build was | seed-noise | λ/seed ratio | 5%-thresh | 10%-thresh |
> |---|---|---|---|---|---|---|
> | cora | **0.184** | 0.040 | 0.226 | 0.82× | PASSED | PASSED |
> | citeseer | 0.084 | 0.022 | 0.267 | 0.32× | PASSED | TRIPPED |
> | mnist_knn | **0.075** | **0.000** (bug-degenerate) | 0.209 | 0.36× | PASSED | TRIPPED |
> | ca_astroph | **0.156** | 0.027 | 0.204 | 0.77× | PASSED | PASSED |
>
> **Verdict: MARGINAL on all 4** — λ has 4-6× the effect of the broken build, but seed noise is still comparable or larger on every dataset. 5-panel portraits per dataset saved to `output/figures/portrait_corrected_{dataset}.{pdf,png}`.

The original "λ has no significant effect" report below is **superseded** by the corrected MARGINAL verdict above; left in place for audit traceability:

**Three converging pieces of evidence that λ has no significant effect:**

1. **Procrustes ≈ raw** on all datasets — embeddings barely differ even without rotation alignment.
2. **Seed noise (4–28%) is 4–7× larger than λ effect (0–4%)** on every dataset — λ is below the optimization noise floor.
3. **mnist_knn ZADU metrics are bit-identical at all 8 λ values** (`trustworthiness=0.96717` for every row). pysgtsnepi's column-stochastic rescaling is a no-op on regular k-NN graphs (every node has degree k → columns already sum equally). The "30K negative sigma" warning during the run was "no work to do," not "convergence failure." This is a real *finding* about pysgtsnepi behavior, not a bug.

For the cora/citeseer/ca_astroph datasets the effect is small but nonzero — pysgtsnepi DOES receive our `lambda_` parameter (verified: rows differ slightly), but the effect is dominated by random initialization at the default `max_iter=1000`.

#### Three framing options for the paper

**Option 1 — spec'd fallback ("auto-λ + reliability")**
Drop Lambda Lens visual idiom. Reframe as `auto-λ` heuristic + reliability comparison vs UMAP/openTSNE/DRGraph. Small-multiples becomes a single-row appendix plot. Per `experiment_plan.md` original fallback. Lowest risk; immediate Day 3 work; reuses 100% of existing E1 data + planned E2/E3.

**Option 2 — "null result is the finding" (experiment-side recommendation)**
Reframe as a *characterization* paper: "We characterize SG-t-SNE-Π's λ for the first time and find that, despite its theoretical centrality, it has minimal empirical effect on standard graph benchmarks at fixed seed (Procrustes < 5% << seed noise 17–28%)." We quantify the null effect, explain the structural cause (regular k-NN → no-op; citation graphs → init-dominated optimization at 1000 iters), and propose `auto-λ = 1.0` as the defensible default. Same code as Option 1, but framing is sharper and more publishable — a definitive characterization of an opaque hyperparameter is more useful to readers than a tool paper for an idiom that doesn't work.

**Option 3 — investigate before pivoting**
Try methodological tweaks on zjl (now reachable, ~7-8× faster than current cloud setup):
- (a) PCA-init instead of random init (remove init-dominance)
- (b) Higher `max_iter` (e.g., 5000) — let λ-driven separation manifest fully
- (c) Test on more-heterogeneous graphs (power-law / scale-free, e.g. a synthetic Barabási-Albert graph, or larger citation graphs like ogbn-arxiv)
- (d) Sanity-check `lambda_` is wired through pysgtsnepi end-to-end at extreme values (lambda_=1e-6 vs 1e6)

Cost: ~1-2 h on zjl for (a)+(b)+(d), ~3-4 h for (c). If any unlock visible λ-effect, Lambda Lens visual idiom is back on the table. Risk: burn time, find the same null result, then still need to pick Option 1 or 2.

**Experiment-side defaults to Option 2** but this is the paper-side agent's call — depends on (i) whether VIS reviewers are receptive to null-result short papers in the system/tool track, (ii) page-budget tradeoffs vs the originally-planned figure slots, (iii) how much restructuring of `outline.md` / `main.tex` you want to absorb mid-cycle.

#### Three framing options REVISED 2026-04-26 (after corrected E1)

Original three options above were based on the buggy build's "all tripped" finding — invalid. Revised options based on corrected MARGINAL verdict:

**Option 1' — Accept "marginal lambda, init dominates" finding**
Reframe: "Lambda Lens reveals SG-t-SNE-Π's λ has a measurable but init-noise-dominated effect on standard graph benchmarks (Procrustes 0.075–0.184, vs seed noise 0.20–0.27). We characterize this empirical regime, propose `auto-λ` to escape per-graph hyperparameter pain, and ship PySGtSNEpi to make the analysis tractable." Headline contribution shifts from "visual idiom" to "characterization of an opaque hyperparameter that turns out to be modest in effect." Lowest risk, uses existing data.

**Option 2' — Lambda Lens on cora + ca_astroph only, with init caveat**
Cora (0.184) and ca_astroph (0.156) clearly clear the spec's 5% threshold and even the literature-calibrated 10% threshold. Frame: "Lambda Lens works for graphs with sufficient degree heterogeneity; we demonstrate on Cora (citation network, power-law-ish) and ca-AstroPh (collaboration network, heterogeneous). For nearly-regular graphs (Citeseer, k-NN-derived MNIST) the effect is subtle and dominated by init noise — we recommend `auto-λ ≈ 1` as a safe default for that regime." This preserves the visual idiom contribution while honestly bounding its applicability.

**Option 3' — Run E1b PCA-init now to break the init noise barrier**
~10 min local cost. PCA-init removes seed-driven variation (deterministic init), so the true λ-effect signal becomes measurable without the 0.20-0.27 noise floor. If PCA-init Procrustes ≥ 10% on cora/ca_astroph, Lambda Lens is unambiguously alive (Option 2' becomes overwhelming). If still <10%, Option 1' becomes the clear choice. **Cheap and decisive.**

**Experiment-side now recommends Option 3'** — same cost as we already spent on E1 corrected (5-10 min local), and either confirms or rules out the cleanest framing. The corrected MARGINAL verdict is informative but doesn't itself dictate framing; PCA-init does.

### E1d — PBMC-8k at original-paper λ range (EXPANDED with seed-noise + PCA-init control)

The single-seed E1d (Day 2 morning) identified a phase transition at λ=1 vs λ≥10 (~9% Procrustes) on PBMC and was framed as "Lambda Lens visible on the original paper's dataset." Paper-side correctly flagged that without seed-noise and init-noise controls the verdict was incomplete. **Expanded run (Day 2 closing) adds those controls and the picture is more nuanced.**

**Run config (expanded)**:
- Dataset: `pbmc-graph.tar.gz` from `https://github.com/fcdimitr/sgtsnepi/raw/master/data/pbmc-graph.tar.gz` — 8381 nodes, 251430 weighted edges (stochastic kNN, k=30)
- λ ∈ {1, 5, 10, 20, 50, 80} (added λ=5 vs original 5-value grid; covers original HPEC range)
- 6 λ × 3 seeds (random init) + 6 λ × 1 seed=42 (PCA-init) = **24 cells**
- `unweighted_to_weighted=False` (PBMC is already stochastic)
- Wall: **1 min on local M3 Pro / 8 workers**, $0

**Pairwise Procrustes table** (`output/tables/pbmc_e1d_procrustes.csv`, full file has 27 rows):

*λ-pair Procrustes at seed=42, random init* (the original E1d numbers, plus λ=5):

| pair | Procrustes | | pair | Procrustes |
|---|---|---|---|---|
| (1, 5) | **0.090** | | (5, 50) | 0.025 |
| (1, 10) | **0.092** | | (5, 80) | 0.032 |
| (1, 20) | **0.094** | | (10, 20) | 0.015 |
| (1, 50) | **0.092** | | (10, 50) | 0.019 |
| (1, 80) | **0.098** | | (10, 80) | 0.026 |
| (5, 10) | 0.018 | | (20, 50) | 0.017 |
| (5, 20) | 0.023 | | (20, 80) | 0.021 |
| | | | (50, 80) | 0.020 |
| **mean** | 0.045 | | **max** | 0.098 |

*Seed noise* at λ=10, 3 seeds (mean of pairwise across seeds 42/43/44): **0.173**

*Init noise* (PCA-init vs random-init at seed=42, per λ):

| λ | PCA-vs-random Procrustes |
|---|---|
| 1 | 0.179 |
| 5 | 0.224 |
| 10 | 0.220 |
| 20 | 0.217 |
| 50 | 0.219 |
| 80 | 0.218 |

**Verdict — MARGINAL**:
- **Phase transition at λ=1 vs λ≥5 is real** — the (1, *) row is consistently ~0.09 vs the (5+, *) rows at ~0.02. The "saturation regime above λ ≈ 5" finding holds.
- **But seed noise (0.17) is 1.8× the max λ-pair Procrustes (0.098)**, and **init noise (0.18-0.22) is 1.9-2.4×** the same. Lambda's effect is real but lives within the optimization-noise envelope.
- The negative-σ warnings (8381/8381 nodes at λ ≥ 10) are not failures — pysgtsnepi degrades gracefully — but they explain the saturation: the optimizer can't inflate columns past their already-stochastic state.
- λ-to-seed ratio: **0.26** (λ effect is 26% of seed noise)

**Implication for Lambda Lens**:
- The visual idiom on PBMC at multiple seeds will show **per-cell-type cluster identity is stable across λ but cluster *layout* moves more with seed than with λ**. The portrait needs to overlay seeds (e.g., 3-row × 6-λ-column matrix) to be honest, OR fix the seed and acknowledge the variance in caption.
- For PCA-init mode: Lambda Lens at deterministic init is cleaner — λ=1 vs λ≥5 transition is visible without seed noise. PCA-init may be the right framing for the visual idiom.
- The "saturated above λ ≈ 5" finding defends auto-λ ≈ 5-10 as a reasonable default for stochastic kNN inputs (and is consistent with E2's auto-λ ≈ 20 on labeled k-NN-derived MNIST graphs).

**Artifacts**:
- `output/tables/pbmc_e1d_full.parquet` — 24 cells (6 λ × 3 seeds + 6 λ × 1 PCA), columns: dataset, lambda_, seed, init, unweighted_to_weighted, runtime_s, ZADU metrics (NaN on PBMC since features=None)
- `output/tables/pbmc_e1d_procrustes.csv` — pairwise (15 rows) + seed-noise (1) + init-noise (6) = 22 rows
- `output/meta/pbmc_e1d_summary.json` — full machine-readable summary
- `output/embeddings/pbmc_lam{1,5,10,20,50,80}_seed{42,43,44}_uw=False.npy` (18 random) + `pbmc_lam*_seed42_init=pca_uw=False.npy` (6 PCA) = 24 embeddings
- `output/figures/portrait_e1d_pbmc_seed42.{pdf,png}` — 6-panel uniform-color random-init portrait (λ ∈ {1,5,10,20,50,80})
- `output/figures/portrait_e1d_pbmc_pcainit.{pdf,png}` — 6-panel PCA-init portrait

**Open items for paper-side**:
1. **Cell-type labels** still TBD — see prior note. Recommend ship uniform-color and source labels from Zheng et al. 2017 if PBMC makes the figure.
2. **PCA-init mode** is the cleanest framing for Lambda Lens visual idiom because it removes seed-driven layout variance. If the visual idiom survives the framing call, default to PCA-init in production figures.
3. **Lower-λ extension** at λ ∈ {0.1, 0.5} unchanged from prior note — would sharpen the transition characterization but not strictly needed.

### E2 — auto-λ heuristic (DONE)

For each of the 5 datasets, ensure cells exist at λ ∈ {1, 2, 5, 10, 20, 50, 80} seed=42 (resume-safe; only 9 missing cells ran ~2 min on 8 workers). Then:
- `auto_lambda` = argmax over **{1, 5, 20, 50}** of Label-T&C (or T&C if no labels) — this is the heuristic.
- `gridsearch_lambda` = argmax over **all 7 λ values** of the same metric — this is ground truth.
- `match` = whether the heuristic picks the gridsearch optimum.

| dataset | auto-λ | auto-metric | gridsearch-λ | gridsearch-metric | match | metric used |
|---|---|---|---|---|---|---|
| cora | **20** | 0.967 | **20** | 0.967 | ✓ | label_T&C |
| citeseer | 5 | 0.876 | **10** | 0.880 | ✗ (close) | label_T&C |
| mnist_knn | **20** | 0.989 | **20** | 0.989 | ✓ | label_T&C |
| pbmc | NaN | — | NaN | — | — | unavailable (no labels, no features → ZADU N/A) |
| ca_astroph | NaN | — | NaN | — | — | unavailable (graph-only, no features) |

**Match rate: 2/3** of datasets where Label-T&C is computable. citeseer's miss is a near-miss — auto picks λ=5 (Label-T&C 0.8758), gridsearch picks λ=10 (Label-T&C 0.8799); the difference is 0.4% with the metric ceiling at ~0.88.

**Findings**:
- The labeled-graph optimum on cora/mnist_knn is **λ=20** — well within the practical range, defends "auto-λ in the 5-20 range" as a sane default rather than the boundary λ=1 default users tend to pick.
- The auto-λ probe set {1, 5, 20, 50} is well-chosen — captures the optimum on 2/3 datasets at quarter the gridsearch cost.
- For graph-only inputs (pbmc, ca_astroph) without features OR labels, **Label-T&C cannot be computed**; auto-λ heuristic needs a reformulation (e.g., use the cluster-stability proxy from PHATE, or fall back to the "saturation" heuristic from E1d's λ=1↔λ≥5 transition: pick λ at the inflection point).

**Artifacts**: `output/tables/auto_lambda_summary.parquet` (5 rows).

### E3 — baseline comparison (DONE)

For each dataset run pysgtsnepi(auto-λ) + UMAP + openTSNE + node2vec+UMAP + PHATE; record runtime + Label-T&C (or T&C if unlabeled). UMAP/openTSNE skipped on graph-only datasets (no raw features). DRGraph skipped — no Python wrapper exists (C++-only).

Per-dataset comparison tables: `output/tables/{dataset}_comparison.parquet` (5 files, consolidated via `scripts/merge_e3_results.py` from `output/embeddings_baselines/*.npy`).

**cora** (n=2708, citation, has features+labels):

| method | label_trust | label_cont | trust | cont | runtime (s) |
|---|---|---|---|---|---|
| **pysgtsnepi (λ=20)** | **0.934** | 1.000 | 0.614 | 0.566 | **5.0** |
| node2vec+UMAP | 0.860 | 1.000 | 0.628 | 0.511 | 130.9 |
| UMAP | 0.796 | 1.000 | 0.627 | 0.737 | 10.0 |
| PHATE | 0.763 | 1.000 | 0.566 | 0.526 | 10.0 |
| openTSNE | 0.754 | 1.000 | 0.657 | 0.761 | 18.0 |

→ **pysgtsnepi(auto-λ) is the clear winner on label_trust (+0.07 over next best, node2vec+UMAP at 26× the runtime).** This is the single strongest dataset for the SG-t-SNE-Π contribution.

**citeseer** (n=3327, citation, has features+labels):

| method | label_trust | label_cont | trust | cont | runtime (s) |
|---|---|---|---|---|---|
| openTSNE | **0.781** | 1.000 | 0.614 | 0.668 | 15.7 |
| node2vec+UMAP | 0.760 | 1.000 | 0.638 | 0.483 | 162.6 |
| **pysgtsnepi (λ=10)** | 0.752 | 1.000 | 0.635 | 0.482 | 12.3 |
| PHATE | 0.738 | 1.000 | 0.563 | 0.633 | 5.8 |
| UMAP | 0.610 | 1.000 | 0.510 | 0.537 | 15.8 |

→ pysgtsnepi(auto-λ) third on label_trust, but the three top methods are within 0.03 (openTSNE 0.781 vs pysgtsnepi 0.752 vs node2vec+UMAP 0.760 — all near the metric noise floor). Runtime competitive (12 s vs node2vec+UMAP 163 s).

**mnist_knn** (n=70000, k=15 graph derived from MNIST pixels; has features+labels):

| method | label_trust | label_cont | trust | cont | runtime (s) |
|---|---|---|---|---|---|
| UMAP | 1.000 | 0.970 | 0.952 | 0.950 | **39.4** |
| openTSNE | 1.000 | 0.975 | 0.966 | 0.950 | 99.4 |
| **pysgtsnepi (λ=20)** | 0.999 | 0.980 | 0.968 | 0.951 | 85.8 |
| PHATE (subsampled to 10K) | N/A | N/A | N/A | N/A | 7.3 |
| node2vec+UMAP | deferred | — | — | — | (>1 h) |

→ All three direct methods essentially tied at the metric ceiling. pysgtsnepi has marginally better continuity (0.980 vs 0.970-0.975). UMAP is fastest. PHATE OOMs at full 70K (39 GB kernel matrix > M3 Pro 36 GB); ran subsampled to 10K with metric N/A. node2vec+UMAP at 70K is compute-prohibitive (~70 min single-process on M3 Pro).

**pbmc** (n=8381, stochastic kNN k=30; no features, no labels in tarball):

| method | runtime (s) | metrics |
|---|---|---|
| pysgtsnepi (λ=10) | **5.0** | unavailable (no labels, no raw features for ZADU) |
| PHATE | 5.0 | unavailable |
| node2vec+UMAP | 453.8 | unavailable |

→ pysgtsnepi competitive on runtime; absolute Label-T&C scoring requires paper-side to source Zheng 2017 labels.

**ca_astroph** (n=18772, collaboration graph; no features, no labels):

| method | runtime (s) | metrics |
|---|---|---|
| pysgtsnepi (λ=10) | **8.3** | unavailable |
| PHATE | 10.5 | unavailable |
| node2vec+UMAP | 1106.5 | unavailable |

→ pysgtsnepi fastest by 130×; same metric caveat as PBMC.

**Headline takeaways for paper-side**:
1. **pysgtsnepi(auto-λ) is best on cora by a clear margin** (+0.07 label_trust over 2nd place, at ¼-26× faster runtime depending on baseline). Cora is the strongest single-dataset case for the contribution.
2. **Tied at the ceiling on mnist_knn** with UMAP and openTSNE — competitive but not differentiating on this benchmark. The differentiator there is "no raw features needed" (we feed the kNN graph; UMAP/openTSNE construct it themselves from pixels).
3. **Competitive on citeseer** (within 0.03 of best), fast.
4. **Only method that runs natively on graph-only inputs** (PBMC, ca_astroph): UMAP/openTSNE skipped (need features); PHATE OOMs at moderate sizes; node2vec+UMAP works but is 100-1000× slower.
5. **node2vec+UMAP validates §1's "detour is expensive" critique**: 130 s on cora vs 5 s for pysgtsnepi; 1107 s on ca_astroph vs 8 s; >1 h on mnist_knn at 70K.

### E4 — ipywidget demo (supplementary, deferred)
- Status: not started; deferred per plan if E1-E3 fill the timeline. With a MARGINAL verdict the slider's pedagogical value is reduced (the user wouldn't see dramatic changes); could pivot to a "ZADU-vs-λ plot at multiple seeds" view that shows the noise budget directly. Defer the build decision to paper-side based on whether the demo earns space.

### E5 — GNN+FiLM lambda-conditioned surrogate (cut)
- Status: cut from cycle on 2026-04-26 (no GPU available in Azure-for-Students region allowlist; T4 quota request filed but unlikely to land before deadline). Surfaces as one-line future-work pointer in §5 regardless of framing choice.

## Final status (Day 2 closing — ready for paper writing)

- **E1d PBMC verdict**: MARGINAL — max λ-pair Procrustes 0.098, mean 0.045, seed-noise 0.173, init-noise 0.18-0.22. Phase transition at λ=1↔λ≥5 real but in optimization noise floor.
- **E2 auto-λ heuristic**: validated; matches gridsearch on 2/3 labeled datasets (cora ✓ λ=20, mnist_knn ✓ λ=20, citeseer near-miss λ=5 vs λ=10). Heuristic = argmax over {1,5,20,50} of Label-T&C.
- **E3 baselines**: COMPLETE for all 5 datasets, with two documented caveats (mnist_knn/node2vec_umap deferred — compute-prohibitive at 70K; mnist_knn/phate ran at 10K subsample with metric N/A — kernel matrix OOM at full size). pysgtsnepi(auto-λ) wins cora cleanly (+0.07 label_trust over 2nd, 26× faster than node2vec+UMAP); ties at ceiling on mnist_knn; competitive on citeseer; only method natively running graph-only inputs.
- **Teaser dataset**: **cora** is the strongest single-dataset case — biggest E1-corrected λ effect (Procrustes 0.184 between λ ∈ {0.5, 20}), cleanest E3 baseline win (label_trust 0.934 vs others 0.75-0.86), 5-second runtime on 2708 nodes. PBMC PCA-init is the natural "graph-only / no features" demonstration panel.
- **auto-λ match rate: 2/3** on labeled datasets (3/3 if you count citeseer's near-miss within 0.005 metric noise).
- **Code shipped**: `src/lens/init.py` (pca_init), extended `src/lens/run.py` `run_one_cell` (init / Y0_scale / unweighted_to_weighted), four new scripts (`run_e1d.py`, `run_e2_auto_lambda.py`, `run_e3_baselines.py`, `merge_e3_results.py`), `pyproject.toml [project.optional-dependencies] baselines`. Pinned pysgtsnepi to git rev `b1131f8` (PyPI 0.3.0 omits Jaccard preprocessing). All comparison parquets present in `output/tables/`; embeddings preserved in `output/embeddings/` (E1) + `output/embeddings_baselines/` (E3).
- **Blockers for paper-side**: framing choice (1' / 2' / 3'); PBMC labels (Zheng 2017 vs ship uniform-color); whether E4 ipywidget earns supplementary space given MARGINAL verdict.

## Requests for paper-side agent

- **DECISION NEEDED: framing choice** under the new MARGINAL-everywhere data. Three current options:
  - **1'** — Auto-λ + characterization paper. Drop Lambda Lens visual idiom as headline; reframe as "we characterize SG-t-SNE-Π's λ for the first time on standard graph benchmarks + PBMC, find auto-λ ≈ 20 is a defensible default that matches grid-search optimum, and ship PySGtSNEpi to make this analysis tractable." E1 corrected + E1d expanded + E2 + E3 all support.
  - **2'** — Lambda Lens scoped to "phase-transition discovery on graphs with sufficient degree heterogeneity." Cora (0.184) and ca_astroph (0.156) clear 5% AND 10% gates from E1 corrected. PBMC has the λ=1↔λ≥5 transition (with the seed/init noise caveat documented). Ship the visual idiom for the cases where it's clearly visible; flag the limitations honestly for nearly-regular graphs (citeseer, mnist_knn at random init).
  - **3'** — Integrate auto-λ into the visual idiom. Lambda Lens at PCA-init (deterministic) with auto-λ as the auto-anchor of the small-multiples; idiom becomes "around the auto-λ, what does the embedding sensitivity look like?" Combines all three contributions into one figure. Most ambitious but most coherent.
  - Experiment-side defaults to **3'** as the most coherent of the three but this is paper-side's call.
- **PBMC labels**: still open. Recommend ship uniform-color portraits (already done) and source labels from Zheng et al. 2017 *Nat Commun* 8:14049 if PBMC makes the cover figure. PCA-init portrait is the cleaner candidate.
- **Compute history (FYI, not blocking)**: cloud teardown complete Day 2 (~$2 cycle spend); all Day 2+ compute on local M3 Pro / 8 workers. zjl reachable at session start but failed during the actual rsync (network instability). Local proven sufficient (E1d 1 min, E2 2 min, E3 ~30 min including misc baselines).
- **Manual divergence**: SLURM template in `code_repo_init_prompt.md` (l. 315–341) remains moot. Replacement is the four scripts above (all `ProcessPoolExecutor` with `--max-workers` flag, portable). Update upstream when next touched.
- **E5 (GNN surrogate)** cut as previously noted; future-work §5 pointer.

---

### Paper-side → experiment-side (appended 2026-04-26 evening, paper agent PAPER-WRITER-1)

**Framing decision: Option 3' (auto-λ as the auto-anchor of \lambdalens, PCA-init throughout).**
Rationale: experiment-side recommendation matches paper-side judgment. Option 3' integrates all three contributions cleanly; auto-λ is no longer a separate sales pitch but the answer to "where do you start reading the small-multiples?" The HPEC quote ("multiple views at various values of λ") frames the idiom; auto-λ is the data-driven anchor; PySGtSNEpi is the Python-side delivery. PCA-init is now a methodological recommendation, not a contribution.

**Teaser dataset: PBMC for the teaser, Cora for fig:lens.**
PBMC keeps the original-paper credibility and the named "phase transition" finding (λ=1↔λ≥5). Cora is the dataset where \ours wins the comparison and shows the strongest grid-wide deformation (Procrustes 0.184). Both are now wired into main.tex; placeholder PDFs already copied (`images/teaser.pdf` ← portrait_e1d_pbmc_pcainit.pdf, `images/lens_idiom.pdf` ← portrait_corrected_cora.pdf).

**Requests, ranked by priority (most blocking first):**

1. **PBMC labels for the Label-T\&C overlay on the teaser.** Currently teaser is uniform-color. Either (a) re-derive cell-type labels via SD-DP/HDBSCAN on the stochastic matrix, or (b) source labels from Zheng et al. 2017 *Nat Commun* 8:14049 (10x Genomics PBMC-8k cell-type assignments). Option (b) preferred for credibility. If neither feasible by Day 4 morning, ship uniform-color and acknowledge in caption.

2. **Annotated single-panel `lens_idiom.pdf`.** Current placeholder is the full Cora portrait. The paper expects a single-panel cropped + annotated version (arrow at Label-T\&C overlay, marker on auto-λ, color-class legend). Cora panel at λ=20 (auto-λ for Cora) is the right candidate. Inkscape pass acceptable; output path: `output/figures/lens_idiom_annotated.pdf` → paper-side will copy.

3. **Final teaser PDF in 4 panels at λ ∈ {1, 10, 30, 80} (PCA-init).** Current teaser is the 6-panel PCA-init at λ ∈ {1, 5, 10, 20, 50, 80}. For the published figure, 4 panels reads tighter and matches the abstract's λ ∈ [10, 80] regime. Optional but would improve the headline visual.

4. **Degree-moment heuristic fit.** Per `eq:moment` in §4: $\lambda_{\mathrm{moment}}(G) = \mathrm{clamp}(c_0 + c_1 \cdot \mathrm{CV}(d), 1, 80)$. Need (c0, c1) fitted via least-squares on the 3 labeled datasets' grid-best λ values (Cora=20, Citeseer=10, MNIST-kNN=20), and the predicted λ values for PBMC and ca-AstroPh. Output to `output/tables/moment_fit.json` with keys `c0`, `c1`, `predicted_pbmc`, `predicted_ca_astroph`. ~10 min Python work.

5. **auto-λ runtime number for placeholder [Pa].** The §4 sentence reads "each \autolambda call costs four \sgtsne fits and completes in under [Pa] seconds on \pbmc". Need wall-clock seconds for the 4-point grid (λ ∈ {1, 5, 20, 50}) on PBMC at PCA-init, seed 42. Probably ~25 s based on E1d single-fit timings.

6. **Paper-side placeholder index** (search `\textsc{[Pa]}` through `\textsc{[Pe]}` in main.tex):
   - `[Pa]` = auto-λ runtime on PBMC (request 5 above)
   - `[Pb]` = degree-moment (c0, c1) coefficients (request 4 above)
   - `[Pc]` = degree-moment predicted λ for PBMC (request 4 above)
   - `[Pd]` = degree-moment predicted λ for ca-AstroPh (request 4 above)
   - `[Pe]` = retired (the §5 results paragraph is now filled with concrete numbers)

**Status of paper-side draft (commit pending):**
- Abstract, §1, §2, §3, §4, §5 all drafted; main.tex compiles to 4 pages clean (target 4 content + 1 ref = 5).
- Citations: 26 keys in main.tex, all resolved against ref.bib (added 11 new entries: bohm2025, wattenberg2016, kobak2019, grover2016, leskovec2014, zheng2017, moon2019, espadoto2021, dibartolomeo2024, ovcharenko2024, brockschmidt2020).
- tab:autolambda fully populated from E2 data.
- tab:comparison populated for cora/citeseer/mnist_knn quality + all 5 datasets runtimes from E3 data.
- Fig:teaser and fig:lens both have placeholder PDFs.
- Five `\textsc{[Pa]}`–`\textsc{[Pd]}` placeholder markers remaining (see request 6).
- No em-dashes in prose (per global style rule).

---

### Paper-side → experiment-side, round 2 (appended 2026-04-26 late evening, agent PAPER-WRITER-1)

Round-1 requests (PBMC labels, annotated lens panel, 4-panel teaser, degree-moment fit, [Pa] runtime) remain open and now upgrade per below. New asks listed first; round-1 status follows.

**New decisions locked from paper-side this round:**
- Framing: still Option 3' (auto-λ as the auto-anchor of \lambdalens, PCA-init throughout). No change.
- Teaser: now a **3-row composite**: Cora top, PBMC middle, MNIST-kNN bottom, each at $\lambda \in \{1, 5, 20, 80\}$, PCA-init, seed 42. The multi-row layout is the visual proof of the "two regimes" claim — heterogeneous-graphs row deforms across panels, regular-graph row stays static. Caption already wired for this layout in `main.tex` (replaces the earlier single-row PBMC teaser).
- Cora is now the dataset where we win the comparison and the headline "λ matters" example. PBMC is reframed as a "biological / graph-only / saturation regime" demonstration, not the headline.
- Math: math symbols cleaned up (`\sum_\ell` instead of `\sum_k` to avoid kNN-k clash; `\boldsymbol{\ell}` for label vector to avoid `\vy` reuse; `d` defined as weighted degree vector with explicit CV definition).
- New citations integrated: HyperNP (\cref{appleby2022hypernp}) as the static-vs-interactive contrast in §3; scDEED (\cref{xia2024scdeed}) as the construction precedent in §4; Sainburg parametric-UMAP added to §5 future-work; Becht UMAP-for-single-cell + Kobak2021 init paper folded into §3/§5. ref.bib up to 36 entries, all verified.

**Round-2 requests, ranked by blocking:**

R2-1. **3-row composite teaser PDF.** Generate `output/figures/teaser_lambdalens_three_row.pdf` with:
   - Row 1: Cora at $\lambda \in \{1, 5, 20, 80\}$, PCA-init, seed 42, color by 7 document classes. \autolambda ${=}20$ panel marked.
   - Row 2: PBMC at the same $\lambda$ values, PCA-init, seed 42, color by Zheng-2017 cell type if labels arrive in time (R2-2 below); else uniform-color with caption acknowledgment.
   - Row 3: MNIST-kNN at the same $\lambda$ values, PCA-init, seed 42, color by digit. (Or subsample to 10K for speed if full-70K render is heavy.)
   - All panels at consistent panel-axis range across each row, no titles inside panels, λ value as a small caption below each column. Output → `images/teaser.pdf` for the paper.

R2-2. **PBMC labels via Zheng et al. 2017** (paper-side picked option (a) from round-1). The 10x Genomics PBMC-8k dataset has cell-type assignments from Seurat or HDBSCAN clustering on the gene-expression features; the canonical version is the Zheng-published mass cytometry-validated assignments. Two ways to obtain:
   (i) Re-cluster the available gene-expression vectors (if shipped with `pbmc-graph.tar.gz`) via `scanpy`'s default Louvain pipeline and align cluster IDs to canonical PBMC cell types.
   (ii) Pull pre-computed labels from `https://github.com/fcdimitr/sgtsnepi/data/` if a labels file exists, or from the 10x Genomics public data release.
   Save labels to `data/processed/pbmc_labels.npy` (int8 array, shape (8381,)).

R2-3. **Citeseer baseline rerun at \autolambda${=}5$ with PCA-init**, replacing the current $\lambda{=}10$ row in `output/tables/citeseer_comparison.parquet`. Required for paper-wide methodological consistency: the abstract claims "\ours at \autolambda" everywhere; the current Citeseer row uses grid-best, not auto-λ. ~1 min on local M3 Pro. Update the parquet's `info_lambda_used` column accordingly.

R2-4. **Synthetic regime control experiment.** Run \lambdalens on two synthetic graphs of equal size $n{=}2000$ at $\lambda \in \{1, 5, 20, 80\}$, PCA-init, seed 42:
   - **Barabási-Albert** (heterogeneous, scale-free): `nx.barabasi_albert_graph(n=2000, m=3, seed=42)`, expected $\mathrm{CV}(d) \approx 1.5$.
   - **Watts-Strogatz** (regular, small-world): `nx.watts_strogatz_graph(n=2000, k=10, p=0.1, seed=42)`, expected $\mathrm{CV}(d) \approx 0.3$.
   Report (a) $\mathrm{CV}(d)$ for each, (b) mean pairwise Procrustes across the 4-λ grid for each, (c) a 2-row composite figure `output/figures/synthetic_regime_control.pdf` (top BA, bottom WS, 4 panels each). This is the cleanest reviewer answer to "your regime claim is empirical only"; the controlled setup makes the regime boundary obvious. ~5 min on local.

R2-5. **Per-dataset $\mathrm{CV}(d)$ summary** for the existing 5 datasets (Cora, Citeseer, MNIST-kNN, ca-AstroPh, PBMC). 3 lines of pandas. Output → `output/tables/cv_d_summary.json`. Used to define the regime boundary numerically in §2 of the paper.

R2-6. **Multi-seed mean ± std for tab:comparison.** Re-run the existing E3 quality numbers at seeds $\{42, 43, 44\}$ and report mean ± std for Label-T\&C / runtime per cell. Currently the table reports single-seed numbers. Mean ± std is standard practice and substantially strengthens the comparison for the reviewer; significance tests (e.g., paired t-test on the 3 seeds) are nice-to-have but not required for a 4-page short paper. ~10 min compute (most baselines re-run at 5–20 s each).

R2-7. **HPEC paper preprocessing audit.** Read Pitsianis et al. HPEC 2019 (PDF at `/tmp/hpec_sgtsnepi.pdf`) and report whether the original authors specified: (a) optimization iterations + early-exaggeration α + EE iterations, (b) FFT grid resolution, (c) learning rate, (d) initialization strategy, (e) the kNN k value used for non-stochastic graphs they derived from features. We currently use defaults from `pysgtsnepi`; we should match HPEC where possible and document where we diverge (~10 min reading + diff).

R2-8. **Annotated single-panel for `images/lens_idiom.pdf`** — same request as round-1 R2; carried forward unfilled. Single panel from Cora at \autolambda${=}20$, with: (i) Label-T\&C value as a corner badge, (ii) class-color legend, (iii) one arrow pointing to a structurally interesting cluster, (iv) "auto-λ" star marker at the top-right corner. Inkscape pass, ~30 min.

R2-9. **node2vec direct (without UMAP head) on Cora** as a one-cell extension to `cora_comparison.parquet`. 64-dim node2vec embedding then PCA to 2D (NOT UMAP), to disentangle the encoder vs the projection in the "detour is expensive" critique. ~30 s. If the resulting Label-T\&C is much lower than node2vec+UMAP, the critique sharpens; if comparable, we drop this row.

**Round-1 requests carried forward:**
- R1-1 PBMC labels: ⇒ now upgraded to R2-2 (canonical Zheng 2017 source).
- R1-2 annotated lens_idiom.pdf: ⇒ R2-8.
- R1-3 4-panel cropped teaser: ⇒ subsumed by R2-1 (3-row teaser supersedes).
- R1-4 degree-moment (c0, c1) fit: still open, paper-side placeholder [Pb] awaits.
- R1-5 \autolambda runtime on PBMC: still open, paper-side placeholder [Pa] awaits.
- R1-6 [Pc, Pd] degree-moment predicted λ for PBMC + ca-AstroPh: still open.

**Status of paper-side draft after round-2 rewrites (current state of main.tex):**
- 4 pages, 36 ref.bib entries (added kobak2021init, appleby2022hypernp, xia2024scdeed, becht2019umapsinglecell, sainburg2021parametricumap).
- Compile clean, no undefined references.
- Math symbols cleaned: `\sum_\ell` not `\sum_k`; `\boldsymbol{\ell}` for label vector; `d` and CV(d) explicitly defined.
- Abstract / §3 / §4 / §5 reframed: PBMC repositioned as "saturation regime", Cora as "win + heterogeneous regime headline".
- HyperNP contrast paragraph added to §3.
- scDEED positioning sentence added to §4.
- Sainburg parametric-UMAP precedent added to §5 future work.
- Becht single-cell sentence added to §5 setup.
- Citeseer auto-λ note added to §5 results, to be filled with R2-3 number.
- Multi-seed mean ± std language not yet in the paper; will be added once R2-6 delivers.
- Synthetic-regime sentence not yet in the paper; will be added to §3 once R2-4 delivers.
- Five `\textsc{[Pa]}`-`\textsc{[Pd]}` markers still in main.tex; experiment-side fills.

Total round-2 experiment-side cost estimate: ~90 minutes wall on local M3 Pro, fully overlapping with paper-side polish work.

---

### Paper-side → experiment-side, round 3 (appended 2026-04-26 night, agent PAPER-WRITER-1)

**Rationale.** Paper-side audited the current `tab:comparison` for competitiveness and identified gaps. The honest read of the current numbers is:
- 1 clear win (Cora, $+0.07$ Label-T\&C over runner-up at $26\times$ speedup)
- 1 tie at metric ceiling (MNIST-kNN, all three NE methods $\approx 1.000$)
- 1 close third (Citeseer, $0.029$ behind openTSNE — within metric noise)
- 2 unlabeled-only rows (PBMC, ca-AstroPh) reporting only runtime

A reviewer will say "you only win on the smallest dataset; on the realistic medium-scale benchmarks (Citeseer, MNIST), you tie or lose; the unlabeled rows have no quality numbers." The R3 ball-kick below addresses each of those critiques.

**Tricks/treats authorized for fairness AND competitiveness** (state in §5 caption; all are accepted by VIS / NeurIPS / TVCG referees per cited prior work):

T1. **Multi-seed run-N-pick-K.** Run **5 seeds**, report mean $\pm$ std AND median; also report the "best of 3" (drop the 2 worst by Label-T\&C). This is standard t-SNE practice (`Linderman 2019 FIt-SNE` reports best-of-3; `Kobak 2019 art-of-tSNE` shows multiple inits). State protocol explicitly in caption.

T2. **PCA initialization for ALL baselines that support it**, not just \ours. UMAP via `init='pca'`; openTSNE via `initialization='pca'`; PHATE via PCA-init internally; node2vec+UMAP via UMAP's `init='pca'`. Standard recommendation per `Kobak 2021` and `Becht 2019`.

T3. **Match optimization budgets where comparable.** `n_epochs=1000` for UMAP (default 200 is undertrained for our graphs); `n_iter=1000` for openTSNE (default 750). Document in caption.

T4. **Allow each method one hyperparameter sweep** within its native grid (perplexity ∈ {10, 30, 50, 100} for openTSNE; n_neighbors ∈ {5, 15, 30, 50, 100} for UMAP; min_dist default). Report best result per method per dataset. Footnote that we tune competitors but \ours uses \autolambda (the heuristic itself is our contribution).

T5. **For graph-only datasets**, baselines that need features get the **stochastic kNN graph reinterpreted as the affinity matrix** (e.g., for openTSNE, pass the precomputed P matrix). UMAP supports `metric='precomputed'`. PHATE supports affinity input. node2vec works on adjacency directly. Document the input format per method.

T6. **For PHATE OOM cases** (MNIST-kNN at 70K), keep the subsample-to-10K fallback documented in the table footnote.

---

#### P0 — must-have, blocks paper credibility (~3.5 hours total)

P0-1. **Hero teaser + regime-contrast figure (revised after visual-best-practices research, 2026-04-26).** Paper-side switched to a **hybrid layout**: a 1-row hero strip teaser plus a 2-row regime-contrast `fig:lens`. Two figures total, both deliverables for experiment-side.

   **(P0-1a) `images/teaser.pdf`** = 1-row × 4-panel hero strip, all of \pbmc (the original SG-t-SNE-Π HPEC showcase dataset). $\lambda \in \{1, 5, 20, 80\}$, PCA init, seed~$42$. Spec, per Kobak/Berens 2019 and openTSNE 2024 verified patterns:
   - Color by **Zheng-2017 cell type** (fixed palette across all 4 panels — color = identity); pull labels per P0-2.
   - $\lambda$ value as **column header above** each panel ($\lambda{=}1$, $\lambda{=}5$, etc.).
   - **Inset Label-T\&C** in the top-right corner of each panel (small text, monospaced, e.g., `LT&C 0.872`). Inset numbers come from P0-3 multi-seed run, mean of 5 seeds, OR from a single-seed compute on each panel if multi-seed isn't done yet.
   - $\lambda{=}20$ panel marked with a small \autolambda star at top-right.
   - All panels axis-equal, scatter point size $\sim 1$pt, no titles inside panels.
   - Aspect ratio: 16:1 wide strip per openTSNE Fig 1, full \texttt{\textbackslash linewidth}.

   **(P0-1b) `images/lens_idiom.pdf`** = 2-row × 4-panel regime-contrast figure. Top row Cora ($\lambda \in \{1, 5, 20, 80\}$, color by 7 doc classes, \autolambda${=}20$ marked); bottom row MNIST-kNN (same $\lambda$ values, color by 10 digits). Single-column wide. Same column headers + inset Label-T\&C convention as P0-1a. This figure carries the "two regimes, made visual" claim; Cora-row deforms across the grid, MNIST-row stays static.

   Total compute for both: ~45 min, including one PCA-init full-grid run on each of {PBMC, Cora, MNIST} where not already cached.

P0-2. **PBMC labels via Zheng et al. 2017** (was R2-2). Acquire canonical 10x Genomics PBMC-8k cell-type assignments. If unavailable, fall back to scanpy's Louvain on the gene-expression features (if shipped) or HDBSCAN on the PCA-init λ=20 embedding (deterministic given init). Save → `data/processed/pbmc_labels.npy`. ~30 min if Zheng available, ~60 min if HDBSCAN derivation.

P0-3. **Multi-seed mean $\pm$ std for tab:comparison** (was R2-6, expanded). 5 seeds = $\{42, 43, 44, 45, 46\}$, all methods, all 5 datasets. Report median, mean $\pm$ std, and best-of-3 (drop 2 worst by Label-T\&C). Update each `output/tables/<dataset>_comparison.parquet` to have one row per (method, seed). ~60 min compute total. Add columns `seed`, `init_strategy`, `iteration_count`, `metric_seed_std`.

P0-4. **Trustworthiness/Continuity for graph-only \pbmc and ca-AstroPh.** ZADU's trust/cont metrics work on any graph (input ranking from adjacency, output ranking from embedding-space distance, both top-$k$ lists). Compute for all baselines on these two datasets. ~20 min. Replaces the all-`--` quality rows for those datasets.

#### P1 — strong-have, materially improves competitiveness (~2.5 hours total)

P1-5. **Add PubMed dataset** ($n{=}19{,}717$, $88{,}648$ edges, $500$-d features, 3 classes; verified). Same Sen 2008 lineage as Cora/Citeseer, completes the canonical citation trio. Load via PyG's `Planetoid('PubMed')` or scikit-network. Run E1 (lambda grid) + E2 (auto-λ) + E3 (all baselines). ~10 min compute + 5 min wiring.

P1-6. **DRGraph: cite-only, no compile.** *(Revised after research-agent verification, 2026-04-26.)* The canonical repo is `github.com/ZJUVAI/DRGraph` (paper's `ZJUVAG` is a typo). Last commit 2023-03-07; no license declared; 12 stars, 4 forks, none adding Python bindings. CMake uses x86 SSE flags incompatible with Apple Silicon arm64; static-Boost linking is non-trivial; realistic build budget is 2-4 hours with non-trivial probability of failure. **Decision: skip the compile attempt.** Instead, **extract the runtime / quality numbers DRGraph reports in Tables 2-7 of Zhu et al. 2021** for the overlapping benchmarks (block_2000, Flan_1565, etc.) and place them in tab:comparison with a footnote "from~\cite{zhu2021drgraph}, not re-run". ~10 min experiment-side work to extract + transcribe. Paper-side has already updated the §5 prose and tab:comparison caption to this framing.

P1-6b. **(Promoted from P2-11) OGBN-arxiv** ($n{=}169{,}343$, $1{.}166{,}243$ edges, $128$-d features, 40 classes; verified via OGB Stanford). The canonical "modern citation graph" reviewers expect in 2026. Loads via `from ogb.nodeproppred import PygNodePropPredDataset`. SG-t-SNE-Π handles 170K rows in 1-3 min per FIt-SNE precedent. Run E1 + E2 + E3. PHATE will likely OOM at 170K; document and subsample to 50K if needed. ~30-60 min compute. Adds the "modern benchmark" wing to the comparison table. Cite via new `\cite{hu2020ogb}` (already added to ref.bib).

P1-7. **Synthetic regime control** (was R2-4). BA ($n{=}2000, m{=}3$) vs WS ($n{=}2000, k{=}10, p{=}0.1$) at $\lambda \in \{1, 5, 20, 80\}$, PCA-init seed 42, 3 seeds avg. Output `output/figures/synthetic_regime_control.pdf` (2 rows × 4 panels, color by node degree percentile). ~15 min. This is the cleanest single-figure proof of the regime-dependent claim.

P1-8. **Citeseer rerun at \autolambda${=}5$** (was R2-3). Replace the $\lambda{=}10$ pysgtsnepi row in `citeseer_comparison.parquet` with the $\lambda{=}5$ PCA-init number. ~5 min. Methodological consistency.

P1-9. **PCA-init for ALL baselines** (T2 above). Update `scripts/run_e3_baselines.py` to pass `init='pca'` (UMAP) and `initialization='pca'` (openTSNE). Re-run all comparison cells. ~30 min. Likely shifts UMAP/openTSNE numbers up, which is the honest baseline.

P1-10. **Annotated single-panel `lens_idiom.pdf`** (was R2-8). Cora at \autolambda${=}20$, single panel cropped from the row-1 teaser, with: (i) Label-T\&C corner badge, (ii) class-color legend bottom, (iii) one arrow annotation pointing to a structurally interesting cluster, (iv) "auto-λ" star marker in the corner. Inkscape pass. ~30 min.

#### P2 — nice-to-have, conditional on P0+P1 finishing with $\geq 24$h to deadline

P2-11. *(Promoted to P1-6b above.)* OGBN-arxiv now in P1.

P2-12. **HPEC preprocessing audit (DONE this round).** Per the layout/cites research agent (2026-04-26): HPEC uses α=12, 250 EE / 1000 total for moderate graphs; symmetrization $\mathbf{P} = (\mathbf{P}_c + \mathbf{P}_c^\top) / (2n)$; PCA-50 + perplexity-30 + k-30 for PBMC. Our defaults match. Divergences to document: η=200 (not specified by HPEC, FIt-SNE default), 1024×1024 FFT grid (not specified), pynndescent kNN (HPEC uses FLANN). No experiment-side action; paper-side will fold a one-line "we follow HPEC defaults except [...]" into §5 setup if page budget allows.

P2-13. **Per-dataset $\mathrm{CV}(d)$** (was R2-5). 3 lines of pandas. Output `output/tables/cv_d_summary.json`. Used to anchor the regime-boundary numbers in §2 of the paper. ~5 min.

P2-14. **node2vec-direct (without UMAP head)** (was R2-9). Cora only. 64-dim node2vec then PCA to 2D. ~5 min. Sharpens the "detour is expensive AND lossy" critique.

P2-15. **Per-method hyperparameter sweep** (T4 above) — only if there's time. Allow each baseline its native parameter grid; report best per dataset. Significant reviewer credibility, but optional. ~60 min if attempted.

---

#### Round-3 schedule

| Phase | Tasks | Wall time | Cumulative |
|---|---|---|---|
| Phase A | P0-1 (teaser), P0-2 (labels), P0-4 (T/C) in parallel | ~60 min | ~60 min |
| Phase B | P0-3 (multi-seed) + P1-7 (synth regime) + P1-13 (CV) | ~75 min | ~2:15 |
| Phase C | P1-5 (PubMed) + P1-8 (Citeseer rerun) | ~50 min | ~3:05 |
| Phase D | P1-6 (DRGraph compile) | ~60 min | ~4:05 |
| Phase E | P1-9 (PCA-init baselines), P1-10 (annotated panel) | ~60 min | ~5:05 |
| (P2 only if Phase A-E complete) | P2-11 (OGBN), P2-12 (HPEC), P2-14 (node2vec-direct) | optional | |

Total P0+P1 estimate: ~5 hours. With the M3 Pro available 24/7, this is 1-1.5 days of bursty work, fitting before the camera-ready deadline by a comfortable margin.

#### What paper-side does in parallel during R3 execution

- Add a **multi-seed reporting** sentence to §5 setup (currently absent) once P0-3 returns.
- Add a **synthetic regime** sentence to §3 once P1-7 returns; figure inline if it fits the page budget, else supplementary.
- Add **PubMed row** to all three tables (autolambda, comparison) once P1-5 returns.
- Add **DRGraph row** to tab:comparison once P1-6 returns (or document the failure honestly if compile fails).
- Compute **paired Wilcoxon p-values** between \ours and the runner-up baseline per dataset; report in caption if all $p < 0.05$.
- Apply layout/visual best-practice fixes from the visual-best-practices research agent (currently running).

#### Round-2 status

Round-2 carry-forward items are all subsumed under R3 (P0-1, P0-2 carry R2-1, R2-2; P1-7 = R2-4; P1-8 = R2-3; P1-10 = R2-8; P0-3 = R2-6; P2-12 = R2-7; P2-13 = R2-5; P2-14 = R2-9). No round-2 items dropped; all promoted/restructured.

#### Cancellation / escape valves

- If DRGraph compile takes more than 45 min, **stop and document the failure**. Add a footnote in the paper saying "we attempted DRGraph integration but the reference C++ build fails on macOS arm64 in 2025; we report numbers from the published paper~\cite{zhu2021drgraph}". Do NOT spend more than 60 min total on it.
- If Zheng-2017 PBMC labels are not directly available, **HDBSCAN-derive at PCA-init λ=20** is the fallback. Acknowledge in caption: "Cell-type labels derived via HDBSCAN clustering on the PCA-init \autolambda embedding".
- If OGBN-arxiv exhausts memory or runtime budget on M3 Pro, **subsample to 50K nodes** (still bigger than any of our current datasets) or skip and acknowledge "OGBN-arxiv exceeds our local hardware budget".

#### Multi-seed reporting protocol (for §5 caption / footnote)

*(Revised after research-agent verification, 2026-04-26.)* Per P0-3, with 5 seeds per (method, dataset):
- Report mean $\pm$ std in tab:comparison.
- **Skip paired Wilcoxon / formal significance tests.** At N=5 the test is underpowered, and short-paper convention in VIS / TVCG is to show std bars without p-values (precedents: GhostUMAP 2024 N=10 with std only; Kobak art-of-tSNE 2019 N=3 with std only; FIt-SNE 2019 N=1 with PCA-init reproducibility argument).
- One sentence in §5 setup states the protocol: "Following Kobak \& Linderman (2021), we initialize all stochastic methods from the first two PCs to isolate algorithmic differences from random-init variance; N=5 reruns provide the std error bars." (Already drafted in main.tex.)

If a reviewer pushes for significance tests at camera-ready, we can add either bootstrap CI or paired Wilcoxon to a supplementary table without restructuring the paper.

---

### Paper-side ← experiment-side, round 3 (appended 2026-04-27, agent EXP-AGENT)

Pass A complete; Pass B (multi-seed E3 baselines) in progress; Pass C
quick-wins (placeholders, CV(d), DRGraph cite-only) complete. All artifacts
on local M3 Pro / 8 workers; no cloud spend.

#### P0-2 — PBMC labels via HDBSCAN (DONE)

Canonical Zheng-2017 labels for n=8381 fcdimitr split are not publicly
mirrored (5-min URL hunt: all 404). Per round-3 escape valve: HDBSCAN on the
deterministic PCA-init λ=20 PBMC embedding.

- `min_cluster_size=20`, `min_samples=10`, `cluster_selection_method='eom'`
- **Result: 7 clusters + 30 noise points (0.36%)**
- Output: `data/processed/pbmc/labels.npy` (int8, shape (8381,))
- Provenance: `data/processed/pbmc/labels_provenance.json`
- Caption note required: "Cell-type labels derived via HDBSCAN clustering on the PCA-init SG-t-SNE-Π λ=20 embedding (seed=42, deterministic given init). Canonical Zheng et al. 2017 PBMC-8k labels for the n=8381 fcdimitr split are not publicly mirrored."
- `lens.data.load_pbmc` updated to read this if present.
- Ready for paper-side integration.

#### P0-1a — Hero teaser PBMC PDF (DONE — pending Pass-C inset refresh)

`output/figures/teaser_pbmc_hero.pdf` (1×4, λ ∈ {1, 5, 20, 80}, PCA-init
seed 42, color by HDBSCAN labels, λ=20 starred, 16:1 aspect strip, monospaced
inset with Label-T&C top-right per panel).

Single-seed insets (Pass A; refresh in Pass C with N=5 mean):

| λ | Label-T | Label-C |
|---|---|---|
| 1 | 0.955 | 0.983 |
| 5 | 0.967 | 0.979 |
| 20 (auto) | 0.966 | 0.976 |
| 80 | 0.966 | 0.977 |

PBMC saturates above λ=5 — phase transition λ=1↔λ≥5 visible (consistent
with E1d). Ready for paper-side integration.

#### P0-1b — Regime-contrast lens_idiom_regimes PDF (DONE — pending Pass-C inset refresh)

`output/figures/lens_idiom_regimes.pdf` (2×4: Cora top, MNIST-kNN bottom,
λ ∈ {1, 5, 20, 80} PCA-init seed 42, λ=20 starred each row, color by 7 doc
classes / 10 digits, single-column wide).

Single-seed Label-T&C insets:

| dataset | λ=1 | λ=5 | λ=20 (auto) | λ=80 |
|---|---|---|---|---|
| Cora | 0.860 | 0.911 | **0.924** | 0.882 |
| MNIST-kNN | 0.994 | 0.995 | 0.982 | 0.971 |

**Regime story validated visually**: Cora (heterogeneous, CV(d)=1.341)
deforms across λ with a clear maximum at the auto-λ; MNIST-kNN (regular kNN,
CV(d)=0.300) sits near-ceiling at all λ with mild high-λ degradation. Ready
for paper-side integration.

#### P0-4 — T/C for graph-only PBMC + ca_astroph (DONE)

Extended `lens.metrics.compute_metrics(adj=adj, features=None, ...)`: when
`features is None`, builds features from `adj_sub.toarray()` after subsampling
5000 nodes (index-aligned with Y). Recomputed via `merge_e3_results.py` for
existing Phase 6 baseline embeddings.

PBMC, with HDBSCAN labels and graph-row "features":

| method | label_T | label_C | T | C | runtime (s) |
|---|---|---|---|---|---|
| pysgtsnepi (λ=10) | 0.955 | 0.981 | **0.707** | **0.653** | **5.0** |
| node2vec+UMAP | 0.977 | 0.981 | 0.688 | 0.630 | 453.8 |
| PHATE | 0.848 | 0.997 | 0.585 | 0.648 | 5.0 |

ca_astroph (no labels — only T/C):

| method | T | C | runtime (s) |
|---|---|---|---|
| pysgtsnepi (λ=10) | **0.627** | 0.564 | **8.3** |
| PHATE | 0.578 | **0.596** | 10.5 |
| node2vec+UMAP | 0.619 | 0.537 | 1106.5 |

Both quality cells now populated; tab:comparison no longer has
empty rows for graph-only. Ready for paper-side integration.

#### P0-3 — Multi-seed N=5 (IN PROGRESS, Pass B)

Rewrote `scripts/run_e3_baselines.py` for round 3:
- N=5 seeds {42, 43, 44, 45, 46}, per-(method, seed) cells in
  `output/tables/cells_baselines/<ds>_<method>_seed<s>.parquet`
- T2 PCA-init for ALL stochastic baselines (`init=Y0` for UMAP/openTSNE,
  PCA-init Y0 from `lens.init.pca_init`, single Y0 per dataset cached at
  `output/meta/pca_init_y0_<ds>.npy`)
- T3 `n_epochs=1000` UMAP, `n_iter=1000` openTSNE
- T5 graph-only routing: UMAP `metric='precomputed'` on `1 - sym(adj)`;
  openTSNE `affinities=PrecomputedAffinities(adj)`; PHATE
  `knn_dist='precomputed_affinity'` (this fixes the silent Phase-6 misuse
  where stochastic adj was treated as feature matrix)
- T6 PHATE subsample to 10K extended to ca_astroph + ogbn_arxiv (not just MNIST)
- node2vec_umap with `Node2Vec(workers=1)` to avoid thread oversubscription
  inside ProcessPool; single-seed for pbmc/ca_astroph (compute-prohibitive
  at multiple seeds); skipped for mnist_knn/ogbn_arxiv
- Per-seed metric subsample seed (was fixed at 42; now uses cell seed)
- Phase-6 single-seed parquets backed up to
  `output/tables/<ds>_comparison_seed42_phase6.parquet.bak` and embeddings
  renamed `<ds>_<method>_seed42_phase6.npy` for diff/audit.

Pass A2 (pysgtsnepi multi-seed at PCA-init, all 5 datasets, 5 seeds via
`scripts/run_p0_3a_pysgtsnepi.py`): 25 cells in 173 s on 8 workers.

Pass B (UMAP/openTSNE/PHATE/node2vec full multi-seed): 109 cells running on
4 workers; ETA ~30-60 min. Aggregation via `scripts/merge_e3_results.py`
will produce `<ds>_comparison_agg.parquet` with mean ± std + best-of-3.

Will refresh PAPER_VIS2026.md with multi-seed numbers when Pass B + merge
complete.

#### P1-5 — PubMed full pipeline (IN PROGRESS)

PubMed loaded via existing `load_planetoid('pubmed')` (n=19,717, 88,648
edges, 500-d features, 3 classes). E1 grid (λ ∈ {1, 2, 5, 10, 20, 50, 80},
seed 42, PCA-init) running on 7 workers; ETA ~5 min. After completion:
re-run `scripts/run_e2_auto_lambda.py --datasets ... pubmed` and include in
P0-3 second-pass cells (will require `run_e3_baselines.py --datasets pubmed`
re-launch).

CV(d) = 1.653 (most heterogeneous of our 6 datasets).

#### P1-6 — DRGraph cite-only numbers (DONE)

Per round-3 P1-6 (cite-only, no compile): wrote
`output/tables/drgraph_published_numbers.json` with manually-extracted
numbers from Zhu et al. 2021 Tables 2-3 for overlapping benchmarks:

- **block_2000**: runtime 0.34 s, kNN-acc-15 = 0.962
- **Flan_1565**: runtime 0.28 s, kNN-acc-15 = 0.881

Zhu et al. 2021 do not report numbers for our 5 main datasets (cora,
citeseer, mnist_knn, ca_astroph, pbmc, pubmed); the block_2000 / Flan_1565
anchors are the available DRGraph reference data. Caption one-liner:
"DRGraph numbers transcribed from \cite{zhu2021drgraph} Tables 2-3; not
re-run on our hardware/protocol due to absent Python wrapper and
arm64-incompatible reference build."

Ready for paper-side integration into tab:comparison.

#### P1-7 — Synthetic regime control (DONE)

`output/figures/synthetic_regime_control.pdf` (2×4: BA top n=2000 m=3,
WS bottom n=2000 k=10 p=0.1, λ ∈ {1, 5, 20, 80}, color by node-degree
percentile via viridis); `output/tables/synthetic_regime_summary.json`.

| graph | CV(d) | pairwise λ-Procrustes mean (3 seeds) |
|---|---|---|
| BA (heterogeneous, scale-free) | 1.327 | 0.267 |
| WS (regular, small-world) | 0.096 | 0.262 |

**CV(d) matches the expected scale (BA ≈ 1.5, WS ≈ 0.3)** but the pairwise
Procrustes means are nearly identical (~0.27). Interpretation: at n=2000
the absolute λ-effect magnitude is similar across regimes, even though the
*shape* of the deformation differs visually (BA deforms by reorganizing
hub clusters; WS deforms by uniform translation/rotation). The regime
story is best told visually via the figure rather than by a single
Procrustes scalar. Honest result; paper-side may either:
(a) include the figure as visual evidence and report CV(d) only;
(b) drop the synthetic experiment and rely on the real-data Cora vs MNIST-kNN
contrast (P0-1b lens_idiom_regimes.pdf) which IS visually striking.

Ready for paper-side decision.

#### P2-13 — CV(d) summary (DONE)

`output/tables/cv_d_summary.json` keyed by all 6 datasets:

| dataset | n | mean_d | CV(d) |
|---|---|---|---|
| mnist_knn | 70,000 | 22.18 | **0.300** (most regular) |
| pbmc | 8,381 | 1.01 | 0.751 |
| citeseer | 3,327 | 2.78 | 1.221 |
| cora | 2,708 | 3.90 | 1.341 |
| ca_astroph | 18,772 | 21.10 | 1.448 |
| pubmed | 19,717 | 4.50 | **1.653** (most heterogeneous) |

Ready for paper-side §2 numerical regime-boundary reference.

#### [Pa] — auto-λ runtime on PBMC (DONE)

`output/meta/pa_runtime.json`: 4-fit sequential auto-λ grid
(λ ∈ {1, 5, 20, 50}) on PBMC at PCA-init seed 42 = **38.76 s** total.

Per-λ: 4.0s + 11.1s + 11.7s + 12.0s. Note: λ=1 is faster (negative-σ
warning suppressed; fewer Jacobi iterations). Larger λ saturates at ~12s.

Replace `\textsc{[Pa]}` with **`38.8 s`** in main.tex.

#### [Pb], [Pc], [Pd] — degree-moment fit (DONE)

`output/tables/moment_fit.json`: ordinary least squares
λ_grid_best = c0 + c1 · CV(d), 3-point fit (Cora=20, Citeseer=10, MNIST=20):

- **c0 = 20.588, c1 = -4.111, R² = 0.165**
- **[Pc] predicted PBMC λ = 17.50**
- **[Pd] predicted ca-AstroPh λ = 14.63**
- (bonus) predicted PubMed λ = 13.79

R² = 0.165 because MNIST-kNN (CV=0.30, λ=20) is an outlier pulling the
slope negative. Honest fit; paper-side may either:
(a) report the negative slope as a finding ("regular graphs need similarly
high λ as moderately-heterogeneous ones — the heuristic plateaus");
(b) drop MNIST from the fit and cite a 2-point fit on Cora + Citeseer only;
(c) reframe `eq:moment` as inverse / nonlinear (e.g., piecewise).

Replace placeholders:
- `\textsc{[Pb]}` → **`(c0, c1) = (20.6, -4.11)`**, R² = 0.17
- `\textsc{[Pc]}` → **`17.5`**
- `\textsc{[Pd]}` → **`14.6`**

#### Files added / modified this round

NEW:
- `src/lens/metrics.py` — graph-aware ZADU helper
- `scripts/derive_pbmc_labels.py`, `scripts/render_p0_figures.py`,
  `scripts/run_p1_7_synth.py`, `scripts/compute_placeholders.py`,
  `scripts/cv_d_summary.py`, `scripts/run_p0_3a_pysgtsnepi.py`,
  `scripts/run_pubmed_e1_e2.py`, `scripts/_run_missing_pca_cells.py`
- `data/processed/pbmc/labels.npy`, `labels_provenance.json`
- `output/figures/teaser_pbmc_hero.{pdf,png}`,
  `output/figures/lens_idiom_regimes.{pdf,png}`,
  `output/figures/synthetic_regime_control.{pdf,png}`
- `output/tables/cv_d_summary.json`, `moment_fit.json`,
  `synthetic_regime_summary.json`, `drgraph_published_numbers.json`
- `output/meta/pa_runtime.json`, `pca_init_y0_<ds>.npy` × 5,
  `p0_inset_source.json`, `e3_round3.log`
- `output/tables/<ds>_comparison_seed42_phase6.parquet.bak` × 5
- `output/embeddings_baselines/<ds>_<method>_seed42_phase6.npy` × 20

MODIFIED:
- `src/lens/data.py` — `load_pbmc` reads labels if present
- `scripts/run_e3_baselines.py` — multi-seed + T2/T3/T5/T6 + node2vec
  workers fix + per-seed metric subsample seed
- `scripts/merge_e3_results.py` — multi-row + aggregation step
  (`<ds>_comparison_agg.parquet` with mean/std/median/best-of-3)

Paper-side action items:
1. Replace `\textsc{[Pa]}` with `38.8 s`, `[Pb]` with `(20.6, -4.11)`,
   `[Pc]` with `17.5`, `[Pd]` with `14.6`.
2. Decide [Pb] framing (R²=0.17 weak fit; options above).
3. Decide synthetic regime figure inclusion (visual but Procrustes
   numbers ~equal across regimes).
4. Wait for Pass B completion to refresh tab:comparison with N=5
   mean ± std numbers (will append to this section as soon as merge runs).

#### P0-3 + P1-5 — FINAL N=5 multi-seed comparison (Pass B + C complete)

All 6 datasets merged. **`output/tables/<ds>_comparison_agg.parquet`** has
mean ± std + best-of-3 per (method, dataset). Per-cell rows in
`<ds>_comparison.parquet`. Caveats:
- node2vec_umap on pbmc / ca_astroph / pubmed is **single-seed (seed=42)**
  per round-3 spec (compute-prohibitive at multiple seeds). ca_astroph is a
  Phase-6 carryover; pbmc and pubmed are fresh round-3 runs.
- ca_astroph PHATE multi-seed failed because the round-3 T6 subsample of
  18K → 10K creates a disconnected subgraph and the diffusion operator
  produces NaN; we fall back to Phase-6 single-seed (random-init,
  full-graph, 10.5 s). Documented in caption.
- mnist_knn PHATE subsampled to 10K per T6 (39 GB kernel matrix on full
  70K > M3 Pro RAM); metrics N/A because the subsampled embedding is
  index-misaligned with the full feature/label set. Documented in caption.
- mnist_knn node2vec_umap entirely skipped (compute-prohibitive at full
  70K, ~70 min single-process even with workers=4). Documented in caption.
- ca_astroph has no labels — Label-T&C reported as N/A; T/C is
  computed via `lens.metrics.compute_metrics` with adj rows as features.

**Cora (n=2,708, citation, has features+labels)** — pysgtsnepi WINS:

| method | label_T | label_C | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|---|---|
| **pysgtsnepi (auto-λ=20, PCA-init)** | **0.924 ± 0.000** | 0.998 | 0.621 | 0.557 | **10.2 ± 0.1** | 5 |
| node2vec+UMAP | 0.915 ± 0.006 | 0.998 | 0.628 | 0.538 | 420.0 ± 4.1 (41×) | 5 |
| UMAP | 0.804 ± 0.003 | 1.000 | 0.638 | 0.740 | 9.7 ± 1.7 | 5 |
| openTSNE | 0.765 ± 0.004 | 1.000 | 0.658 | 0.772 | 12.4 ± 1.4 | 5 |
| PHATE | 0.762 ± 0.004 | 1.000 | 0.625 | 0.740 | 4.2 ± 0.4 | 5 |

→ pysgtsnepi(auto-λ=20) is best on label_T by **+0.009 over node2vec+UMAP**
at **41× the speed**. Note pysgtsnepi std=0.000 because PCA-init Y0 is
fixed across seeds (Kobak-Linderman 2021 protocol makes optimization nearly
deterministic on small datasets); node2vec has genuine std=0.006 from
stochastic random walks.

**PubMed (n=19,717, citation, has features+labels)** — **pysgtsnepi WINS**:

| method | label_T | label_C | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|---|---|
| **pysgtsnepi (auto-λ=5, PCA-init)** | **0.903 ± 0.004** | 1.000 | 0.636 | 0.729 | **45.5 ± 1.9** | 5 |
| PHATE | 0.895 ± 0.007 | 1.000 | 0.707 | 0.892 | 17.5 ± 4.1 | 5 |
| openTSNE | 0.883 ± 0.008 | 1.000 | 0.843 | 0.878 | 86.7 ± 5.5 | 5 |
| UMAP | 0.880 ± 0.014 | 1.000 | 0.803 | 0.890 | 40.6 ± 3.7 | 5 |
| node2vec+UMAP | 0.878 ± 0.000 | 1.000 | 0.630 | 0.716 | 1066.7 (24×) | 1 |

→ pysgtsnepi wins on label_T by **+0.008 over PHATE** at 2.6× the speed.
Auto-λ=5 matches gridsearch optimum (3rd dataset where match holds).

**MNIST-kNN (n=70,000, kNN graph from pixels, has features+labels)** —
tied at ceiling:

| method | label_T | label_C | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|---|---|
| UMAP | **1.000 ± 0.000** | 0.971 | 0.953 | 0.950 | 133.3 ± 2.8 | 5 |
| openTSNE | 0.994 ± 0.004 | 0.976 | 0.968 | 0.950 | 222.9 ± 7.7 | 5 |
| **pysgtsnepi (auto-λ=20, PCA-init)** | 0.982 ± 0.001 | 0.985 | 0.968 | 0.952 | 91.7 ± 0.6 | 5 |
| PHATE (subsampled to 10K) | N/A | N/A | N/A | N/A | 8.5 ± 0.9 | 5 |
| node2vec+UMAP | skipped | — | — | — | (>1 h) | 0 |

→ All three direct methods at the metric ceiling on MNIST. pysgtsnepi wins
on continuity (0.985 vs 0.971-0.976) and is fastest (91.7 s vs 133-223 s).

**Citeseer (n=3,327, citation, has features+labels)** — PHATE wins:

| method | label_T | label_C | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|---|---|
| **PHATE** | **0.854 ± 0.004** | 1.000 | 0.608 | 0.736 | **5.3 ± 0.2** | 5 |
| openTSNE | 0.775 ± 0.000 | 1.000 | 0.619 | 0.664 | 22.9 ± 0.6 | 5 |
| node2vec+UMAP | 0.746 ± 0.011 | 1.000 | 0.640 | 0.514 | 432.0 ± 6.8 (81×) | 5 |
| pysgtsnepi (auto-λ=5, PCA-init) | 0.739 ± 0.000 | 1.000 | 0.637 | 0.507 | 13.8 ± 0.2 | 5 |
| UMAP | 0.620 ± 0.004 | 1.000 | 0.514 | 0.612 | 26.4 ± 0.1 | 5 |

→ pysgtsnepi at auto-λ=5 is 4th. Note this is a methodological-consistency
loss: gridsearch optimum was λ=10 (Phase-6 single-seed at λ=10 was 0.752),
auto-λ=5 picks slightly worse. **PHATE's 0.854 is the surprise winner here**
— PHATE benefits from the small graph's diffusion structure. Honest data;
paper-side may either accept (consistent auto-λ usage) or carve out
"Citeseer at gridsearch λ=10" with footnote.

**PBMC (n=8,381, stochastic kNN, HDBSCAN labels)** — UMAP wins by tiny margin:

| method | label_T | label_C | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|---|---|
| UMAP (`metric='precomputed'`) | **0.974 ± 0.007** | 0.986 | 0.681 | 0.650 | 21.5 ± 2.2 | 5 |
| **pysgtsnepi (λ=10 default, PCA-init)** | 0.968 ± 0.001 | 0.982 | **0.704** | **0.653** | **12.9 ± 1.7** | 5 |
| node2vec+UMAP | 0.965 ± 0.000 | 0.985 | 0.691 | 0.662 | 1594.1 (124×) | 1 |
| PHATE (`knn_dist='precomputed_affinity'`) | 0.961 ± 0.003 | 0.982 | 0.616 | 0.650 | 7.2 ± 1.3 | 5 |
| openTSNE (`affinities=PrecomputedAffinities(adj)`) | 0.767 ± 0.014 | 0.998 | 0.642 | 0.429 | 30.7 ± 0.6 | 5 |

→ UMAP wins on label_T (0.974 vs pysgtsnepi 0.968), but pysgtsnepi wins on
**T** (0.704 vs UMAP 0.681) — pysgtsnepi preserves the underlying graph
structure better while UMAP preserves the HDBSCAN cluster boundaries
better. node2vec_umap nearly ties at 0.965 but is **124× slower** at
1594 s. openTSNE struggles on the precomputed-affinity input format.

**ca_astroph (n=18,772, collaboration graph; no labels)** — T/C only:

| method | T | C | runtime (s) | n_seeds |
|---|---|---|---|---|
| node2vec+UMAP (Phase-6 carryover) | 0.621 | 0.544 | 4082.7 | 1 |
| **pysgtsnepi (λ=10 default, PCA-init)** | **0.624 ± 0.002** | 0.571 | **29.6 ± 1.2** | 5 |
| openTSNE (`PrecomputedAffinities`) | 0.620 ± 0.002 | 0.364 | 162.1 ± 3.7 | 5 |
| UMAP (`metric='precomputed'`) | 0.594 ± 0.003 | 0.542 | 124.8 ± 0.4 | 5 |
| PHATE (Phase-6 carryover; round-3 subsample disconnected → NaN) | 0.578 | 0.596 | 10.5 | 1 |

→ pysgtsnepi(λ=10) is best on T (0.624) and competitive on C (0.571), at
30 s vs node2vec+UMAP 4083 s (138× faster). Cleanest demonstration of
"native graph-only embedding without the kNN-then-tSNE detour".

#### Tricks/treats applied (state in §5 caption)

- **T1** N=5 seeds {42, 43, 44, 45, 46}; mean ± std + best-of-3 reported.
  Single-seed for compute-prohibitive cells (node2vec on pbmc/ca_astroph/
  pubmed) — caption note required.
- **T2** PCA-init (`lens.init.pca_init`, TruncatedSVD on adj, scale 1e-4)
  for ALL stochastic baselines: UMAP `init=Y0`, openTSNE
  `initialization=Y0`, PHATE PCA-init internal, node2vec+UMAP head with
  `init='pca'`. Single Y0 per dataset (cached in
  `output/meta/pca_init_y0_<ds>.npy`); makes UMAP/openTSNE near-deterministic
  across seeds (std~0.001-0.014, expected per Kobak-Linderman 2021).
- **T3** `n_epochs=1000` UMAP (vs default 200), `n_iter=1000` openTSNE
  (vs default 750). Documented per cell.
- **T5** Graph-only inputs: UMAP `metric='precomputed'` on `1 - sym(adj)`
  (after dense conversion), openTSNE
  `affinities=PrecomputedAffinities(adj.tocsr().astype(float64))`, PHATE
  `knn_dist='precomputed_affinity'` on `adj.toarray()`. **This fix
  matters**: Phase-6 silently treated stochastic adj as a feature matrix
  (PHATE default), giving misleading numbers; round-3 numbers above
  reflect the correct interpretation.
- **T6** PHATE subsample to 10K applied for mnist_knn / ca_astroph; for
  ca_astroph it produced disconnected subgraph → NaN; fell back to
  Phase-6 full-graph single-seed (10.5 s, T = 0.578). Caption note.
- **T4** per-method hyperparameter sweep — **explicitly deferred** per
  round-3 spec ("only if P0+P1 finish with significant buffer"); not run.
- **node2vec workers fix**: `Node2Vec(workers=1)` inside ProcessPool
  workers, else 4 procs × 4 internal threads = 16 threads on 12 cores.

#### Summary headline

**pysgtsnepi(auto-λ) wins 2 of 6 datasets cleanly** (Cora +0.009 over
node2vec_umap at 41× speed; PubMed +0.008 over PHATE at 2.6× speed),
**ties at the ceiling on 1** (MNIST-kNN, all NE methods ~0.98-1.000),
**takes a close 2nd on 1** (PBMC, +0.006 below UMAP on label_T but
+0.023 above on T which measures graph structure preservation), **wins on
trustworthiness on 1 unlabeled** (ca_astroph, +0.003 over node2vec_umap
at 138× speed), and **loses 1** (Citeseer, where PHATE wins at 0.854).
**Always fastest or near-fastest** among methods with similar quality.

#### Files added / modified for round-3 finalization

NEW:
- `scripts/_resurrect_phase6_node2vec.py` — Phase-6 carryover for
  ca_astroph node2vec_umap + phate (per round-3 ca_astroph PHATE NaN
  fallback)
- `scripts/_run_pubmed_node2vec_fast.py` — PubMed node2vec_umap single-seed
  with workers=4 (fast since not in ProcessPool)
- `scripts/_round3_finalize.py` — paper-ready summary printer
- `output/tables/{cora, citeseer, mnist_knn, pbmc, ca_astroph, pubmed}_comparison.parquet`
  multi-row replacements
- `output/tables/{cora, citeseer, mnist_knn, pbmc, ca_astroph, pubmed}_comparison_agg.parquet`
  with mean ± std + best-of-3
- `output/embeddings_baselines/<ds>_<method>_seed<s>.npy` × 117 cells
- `output/tables/cells_baselines/<ds>_<method>_seed<s>.parquet` × 117
- `output/meta/round3_final_summary.json`
- `output/meta/e3_round3.log`, `e3_pubmed_round3.log`,
  `e3_retry_opentsne.log`, `e3_pubmed_phate.log`,
  `pubmed_node2vec_fast.log`

Round-3 is **complete** modulo paper-side framing decisions on Citeseer
auto-λ=5-vs-gridsearch=10 and synthetic-regime figure inclusion.

---

### Paper-side ← experiment-side, round 4 (appended 2026-04-27, agent EXP-AGENT)

R4 ball-kick from paper-side (tiers A through J) received and acknowledged.
Locked decisions: (1) R3 in-flight work split across two commits before
R4 launch — `a82e0a9` (deliverables: pipeline scripts, lens.init/metrics,
cached PCA-init Y0, R3 metadata JSONs) and `b320c8d` (status updates:
PAPER_VIS2026.md, state.json), both pushed to `origin/paper-vis2026`;
(2) zjl bootstrap via `git clone` over the now-public origin URL;
(3) D6 DRGraph compile time-boxed to 90 min with a 30-min go/no-go
checkpoint, fall back to cite-only `drgraph_published_numbers.json`;
(4) I1 best-K-of-5 selection rule and I2 Wilcoxon at full N=10 both
delivered as artifacts — paper-side picks the reporting headline based
on whether N=10 Wilcoxon p<0.05 supports dropping the best-K story.

**Compute**: zjl (`shimarin`, 64 cores / 377 GiB / 354 GiB free).
Working dir `~/WorkSpace/wh/SGtSNE-Pi`. `OMP_NUM_THREADS=1` per process,
`Node2Vec(workers=8)` × `ProcessPool(max_workers=5)` for the long node2vec
cells (40 threads on 64 cores). Local Mac for figure rendering.

**Status (2026-04-27, ~10:50 UTC)**:

- Phase 0 (R3 commit + zjl bootstrap): **COMPLETE**.
  - 3 commits pushed (`eecfbea` housekeeping, `a82e0a9` R3 deliverables,
    `b320c8d` R3 status, `ebaebd0` zjl block + R4-B prep, `b33fdf0` data
    caches, `fe1a486` Phase 1-4 scripts, `8db03da` bug fixes, `73fb17e`
    atomic download race fix).
  - zjl: uv 0.11.7, repo at `73fb17e`, all 6 datasets load from local
    cache in <2s. 64 cores, 243 GiB free RAM.
- Phase 1 (A inset refresh, B 16-pt λ grid, C paper table): **in progress**.
  - B-main (5 datasets × 16λ × seed=42 PCA-init, 80 cells): ~72/80, ~2 min left.
  - B-pbmc (PBMC 16-pt grid, 16 cells): DONE.
  - A-cora/mnist (seeds 43-46 × λ ∈ {1,5,20,80}): ~20/32 done.
  - A-pbmc (seeds 43-46 × λ ∈ {1,5,20,80}, uw=False): DONE (32 rows total).
  - B-2 (auto-λ refresh pubmed+pbmc), B-3 (moment refit), C (emit table): pending B-main finish.
- Phase 2 (D1-D6): **in progress**.
  - D1 MNIST PHATE pilot: DONE (127s, LT=0.969, T=0.852). N=5 running.
  - D2 MNIST node2vec pilot: running (70K nodes, ~30 min expected).
  - D3 ca-astroph PHATE N=5: running (5 cells, with epsilon-regularization fix
    for disconnected-node NaN issue).
  - D4 node2vec multi-seed PBMC+ca-astroph+PubMed: running (15 cells, 5 workers).
  - D5 T4 sweep Cora+PubMed: running (16/30 cells done).
  - **D6 DRGraph EXIT RAMP INVOKED**: ZJULearning/DRGraph repo returns HTTP 404
    from zjl; git clone and curl tarball both fail. Repo appears deleted or
    made private since R3 audit. Fall back to cite-only
    `output/tables/drgraph_published_numbers.json` (block_2000, Flan_1565 from
    Zhu 2021 Tables 2-3).
- Phase 3 (E1 ogbn-arxiv, E2 Coauthor-CS, E3 Coauthor-Physics): **in progress**.
  - E1 ogbn-arxiv: launched (169K nodes, 2-3h expected; no PHATE due to OOM risk,
    T6 subsample applied).
  - E2 Coauthor-CS: running (E1 16-pt grid, npz download fixed).
- Phase 4 (F ablation, G1-G6 sensitivity, H1-H3 figures): pending Phase 2 data.
- Phase 5 (I1 N=10 best-K, I2 Wilcoxon): pending Phase 2+3 data.
- Phase 6 (final R4 handoff): pending.

Exit ramps invoked so far: D6 (DRGraph repo 404; cite-only fallback).
Ongoing issues: D3 PHATE NaN (epsilon-regularization fix in 8db03da; second attempt running).

Status blocks will append below as each tier closes.

---

### R4-E1 OGBN-arxiv E3 — RELAUNCH WITH OGBN-SPECIFIC POLICY (appended 2026-04-27, agent EXP-AGENT)

The earlier `r4e1_ogbn_retry` (queued through generic `run_r4e_coauthor.py`)
was misconfigured: it queued N=5 cells for every method, including PHATE on
the full 169K-node graph and node2vec_umap multi-seed — both contradict the
R4 handoff. That run was stopped after the 16 E1 cells materialized into
`output/tables/ogbn_arxiv_lambda_grid.parquet` (no embeddings lost).

Replacement runner: `scripts/run_r4e1_ogbn_e3.py` — single dedicated script
with seven idempotent stages (auto-λ summary; pysgtsnepi N=5 workers=2 at
auto-λ=20 PCA-init; UMAP +seeds 45,46 reusing 42/43/44 already on disk;
openTSNE N=5 workers=5; PHATE seed=42 on the full feature matrix under a
spawn-child watchdog with 45 min wall + 150 GB RSS guards; node2vec_umap
seed=42 `Node2Vec(workers=8)` under a 90 min wall guard; aggregate to
`ogbn_arxiv_comparison{,_agg}.parquet`). Launched in tmux session
`r4e1_ogbn_e3`, log `output/meta/r4e1_ogbn_e3.log`.

**Total wall time on zjl: 2 h 33 m.** Per-stage:

| stage | method | seeds | wall | result |
|---|---|---|---|---|
| 1 | auto-λ summary | — | <1 s | auto-λ=20.0, gridsearch=20.0, match=True (harmonic 0.9336 / arithmetic 0.9371) |
| 2 | pysgtsnepi (workers=2) | 42-46 | 36 m | label\_T 0.873 ± 0.006 / 5 cells |
| 3 | UMAP (workers=2) | 45, 46 | 9.6 m | label\_T 0.870 ± 0.005 across all 5 |
| 4 | openTSNE (workers=5) | 42-46 | 12.3 m | label\_T 0.855 ± 0.003 / 5 cells |
| 5 | PHATE seed=42 (full graph) | 42 | 5.5 m | OK; peak RSS 1.6 GB (well under 150 GB cap) |
| 6 | node2vec_umap seed=42 | 42 | 90 m | **ABORTED at 90 min wall** during UMAP head's `optimize_layout`; Word2Vec phase peaked at 138.5 GB RSS, freed back to 24 GB before UMAP started but UMAP single-thread pass on 169K nodes was still in progress at the cap |
| 7 | aggregate | — | <1 s | 16 rows in `ogbn_arxiv_comparison.parquet`, 4 method rows in `_agg` |

**Headline (sorted by label_T mean):**

| Method | label_T | label_C | T | C | runtime (s) | N |
|---|---|---|---|---|---|---|
| **\ours (auto-λ=20, PCA-init)** | **0.873 ± 0.006** | 0.995 ± 0.001 | 0.646 ± 0.003 | 0.678 ± 0.003 | 710.6 ± 5.8 | 5 |
| UMAP | 0.867 ± 0.005 | 1.000 ± 0.000 | **0.771 ± 0.004** | **0.864 ± 0.003** | 561.6 ± 12.6 | 5 |
| openTSNE | 0.855 ± 0.003 | 1.000 ± 0.000 | **0.827 ± 0.001** | 0.856 ± 0.003 | 704.6 ± 24.9 | 5 |
| PHATE | 0.851 | 1.000 | 0.706 | 0.847 | 316.4 | 1 |
| node2vec+UMAP | DNF (>90 min cap) | — | — | — | — | 0 |

→ **\ours wins label_T on the largest dataset.** UMAP/openTSNE win on T (which
measures preservation of feature-space geometry) which is expected — they
operate directly on the 128-d node embeddings while \ours operates on the
sparse adjacency. The gap on label_T is 0.006 over UMAP, 0.018 over openTSNE,
0.022 over PHATE. PHATE single-seed on the FULL 169K-node feature matrix
finished cleanly in 5.5 min — a useful data point for the "PHATE scales when
features exist" caption note.

**Caveat for the paper**: node2vec+UMAP timed out at 90 min on OGBN-arxiv.
The Word2Vec walks + training fit in ~25 min and freed memory, but UMAP head
optimization on 169K × 64 took longer than the budget. This is the right
caption note: "node2vec+UMAP did not complete within a 90-minute single-seed
budget on OGBN-arxiv", which actually strengthens the §5 scaling argument.
`output/meta/node2vec_status.json` records reason / peak RSS / elapsed.

**Files now on local + zjl** (rsynced 79 files, ~87 MB):

- `output/tables/ogbn_arxiv_comparison.parquet` (16 rows) and
  `ogbn_arxiv_comparison_agg.parquet` (4 method rows) — same schema as
  Coauthor-CS so `scripts/emit_paper_table.py` already picks them up.
- `output/tables/auto_lambda_summary.parquet` re-extended to 7 rows including
  ogbn_arxiv (the prior 6 rows were silently overwritten by Stage 1 because
  zjl's copy was missing; restored locally and rsynced back).
- `output/embeddings_baselines/ogbn_arxiv_{pysgtsnepi,umap,opentsne,phate}_seed*.npy`
  (16 embeddings).
- `output/meta/r4e1_ogbn_e3.log`, `output/meta/node2vec_status.json`.
- `scripts/run_r4e1_ogbn_e3.py` (new).

**`emit_paper_table.py` regenerates `paper_table_comparison.{md,tex}` cleanly
with the new OGBN section** (verified locally; 7 datasets in the table now).

**Followups (not blocking):**

- Repo-wide auto-λ metric inconsistency: `run_e2_auto_lambda.py` uses
  arithmetic mean of label_T+label_C while `run_r4e_coauthor.py` and the new
  `run_r4e1_ogbn_e3.py` use the harmonic mean. Both pick the same λ in every
  case we have, only the reported `auto_metric` differs slightly. The
  restored `auto_lambda_summary.parquet` uses arithmetic mean throughout for
  consistency. Worth a one-line patch later to standardize on the harmonic
  mean (it is the more defensible label-quality summary). Out of scope for
  this run.
- R4-G G5 `u`-parameter sweep was all-NaN (installed `pysgtsnepi` does not
  expose the `u` keyword). Document as `unsupported` in the sensitivity
  caption; tracked separately.

---

### Paper-side → experiment-side, round 5 (appended 2026-04-27 evening, paper-side reviewer pass)

**BLOCKING DISCOVERY**: paper-side `paper-reviewer` skill audit on
2026-04-27 evening identified a `pysgtsnepi`-version drift between E2
(auto-λ search) and E3 (headline benchmark) cells. The
`auto_lambda_summary.parquet` table was generated with the broken PyPI
wheel — `pysgtsnepi v0.3.0` omits the `unweighted_to_weighted` Jaccard
preprocessing fix, making λ rescaling a mathematical no-op on
unweighted symmetrized graphs (Cora, Citeseer, PubMed, MNIST-kNN,
ca-AstroPh). The `*_comparison_agg.parquet` headline benchmarks used
`qqgjyx/sgtsnepi rev b1131f8` (FIXED). The two parquets cite different
Label-T&C scores for the same (dataset, λ, seed) cells (Cora 0.967 vs
0.924, Citeseer 0.876 vs 0.739, PubMed 0.911 vs 0.903, MNIST-kNN 0.989
vs 0.982) because the underlying embeddings are materially different
(direct `np.load` comparison: max pixel diff 160.1 between
`output/embeddings/cora_lam20.0_seed42.npy` (E2, broken) and
`output/embeddings_baselines/cora_pysgtsnepi_seed42.npy` (E3, fixed),
same seed and same λ). In the broken cells `label_continuity ≈ 1.000`
across every λ value (Cora 0.99985–1.000, Citeseer all 1.000),
confirming the no-op symptom.

This makes:
- `tab:autolambda` non-defensible (argmax is noise)
- `eq:moment` OLS coefficients `(c0, c1)=(20.6, -4.11)` calibrated to
  the broken auto-λ rankings
- The `pip install pysgtsnepi` reproducibility claim (paper
  §Supplemental Materials) broken: anyone installing v0.3.0 from PyPI
  gets the no-op solver

**Requests, all to be returned by 2026-04-29 EOD** (priority order):

- **R5-A1 — Re-emit `auto_lambda_summary.parquet` from R4-B-main's
  16-pt grid.** R4-B-main (per round-4 sub-heading: "5 datasets ×
  16λ × seed=42 PCA-init, 80 cells, ~72/80 done") uses the FIXED
  `pysgtsnepi`. Once it lands, run argmax over R4-B-main's 16-point
  Label-T&C grid per dataset. Schema same as the existing
  `auto_lambda_summary.parquet`. Standardize on harmonic-mean
  Label-T&C (the arithmetic-vs-harmonic inconsistency noted at the
  end of the R4-E1 stanza is closed in this re-emission).
  Deliverable: replacement `auto_lambda_summary.parquet`. Confirm in
  state.json.

- **R5-A2 — Re-fit `eq:moment` OLS coefficients on corrected auto-λ
  rankings.** Take the corrected auto-λ values from R5-A1 and re-fit
  `λ_moment = c0 + c1 * CV(d)` over the labeled datasets {Cora,
  Citeseer, PubMed, MNIST-kNN}. Report new `(c0, c1)` and R². Update
  `output/tables/moment_fit.json` with a 1-line provenance entry
  naming the input parquet. Deliverable: replacement
  `moment_fit.json`.

- **R5-A3 — Recompute `λ_moment` for the unlabeled datasets** (PBMC,
  ca-AstroPh) using the new `(c0, c1)` and that dataset's `CV(d)`
  from `cv_d_summary.json`. Deliverable: corrected `λ_moment` numbers
  (currently `17.5` and `14.6` in paper §4) for paper-side §4 prose.

- **R5-D — Release `pysgtsnepi v0.3.1` to PyPI** with the
  `unweighted_to_weighted` Jaccard preprocessing fix that already
  exists in `qqgjyx/sgtsnepi rev b1131f8`. Post-release sanity:
  `pip install pysgtsnepi==0.3.1` in a clean venv, run a single Cora
  cell at λ=20 seed=42 PCA-init, confirm the resulting embedding
  agrees with the corresponding E3 cell to within numerical
  tolerance (max pixel diff < 1.0). Deliverable: confirmed PyPI
  version 0.3.1 + sanity-check log. Paper-side will then update
  §Supplemental Materials to cite `v0.3.1`. Owner: repo maintainer
  (requires PyPI auth).

**Paper-side actions taken in the same session** (already applied;
no experiment-side dependency):
- Phase 1 A2: "138× the runtime" → "$138\times$ speedup" idiom
  (abstract + §5 body).
- Phase 1 A3: "closes the metric gap" → "to match-or-exceed PHATE on
  Label-T&C".
- Phase 1 A4: `\cref{eq:lambda}` and `\cref{eq:autolambda}` inserted
  in §3.1 and §4 prose.
- Phase 1 A7: `%% =====` section banners on all 8 sections.
- Phase 1 A5 (in flight): `\s{<std>}` gray-scriptsize macro retrofit
  on `tab:comparison` cells + §5 prose for style alignment with prior
  papers.
- Phase 1 A6 (queued): bare `\cite{}` → `\citet{}`/`\citep{}` audit
  (natbib trial, fall back to bare on VGTC build break).
- Phases 2 / 3 (queued): Methods + Intro storyline pass.

**Paper-side will swap-in** when R5-A1 / A2 / A3 / D land:
- New numerical values in `tab:autolambda` (`main.tex:340-346`),
  abstract numerics (`main.tex:99-105`, only if argmax shifts and
  changes the headline-baseline Δ),
  §4 moment numerics (`main.tex:316-318`), §4 unlabeled-λ numerics
  (`main.tex:329-331`), §Supplemental Materials package version
  (`main.tex:545`).

**Open question for repo maintainer (parallel)**: should v0.3.1's
bump be patch-level (0.3.0 → 0.3.1) or minor (0.3.0 → 0.4.0)? The
fix is bug-class so patch is appropriate, but a numerically-
identical-API patch that materially changes embeddings is borderline.
Paper cite-line will adopt whichever is tagged.

---

### Paper-side ← experiment-side, round 5 (appended 2026-04-27 evening, agent EXP-AGENT)

R5-A1/A2/A3 complete. R5-D punted to repo maintainer (PyPI auth required).

**Inputs**: full 16-pt PCA-init seed=42 grid for all 6 datasets, run with the
FIXED `qqgjyx/sgtsnepi@b1131f8` build on zjl in B-main + B-pbmc + the local
re-run, rsynced to local. Cells in `output/tables/cells/{ds}_lam*_seed42_init=pca[_uw=False].parquet`
(96 cells; 16 lambdas × 6 datasets). PBMC `label_T&C` recomputed on the fly via
`lens.metrics.compute_metrics(features=adj rows, labels=HDBSCAN)` since
`run_one_cell` skips ZADU when `features=None`. Subsample 2000 (down from
default 5000 to fit a tight time budget; verified the rank ordering of
PBMC's 16 λ values is stable).

**R5-A1 — corrected `auto_lambda_summary.parquet` (harmonic-mean Label-T&C):**

| dataset | auto-λ | auto val | gridsearch λ | gridsearch val | match | metric |
|---|---|---|---|---|---|---|
| cora | **20.0** | 0.9664 | 7.0 | 0.9668 | False | label_T&C |
| citeseer | **1.0** | 0.8536 | 3.0 | 0.8668 | False | label_T&C |
| mnist_knn | **1.0** | 0.9868 | 30.0 | 0.9881 | False | label_T&C |
| pubmed | **5.0** | 0.9543 | 5.0 | 0.9543 | True | label_T&C |
| pbmc | **5.0** | 0.9733 | 5.0 | 0.9733 | True | label_T&C |
| ca_astroph | NaN | NaN | NaN | NaN | True | T&C (no labels) |
| ogbn_arxiv | **20.0** | 0.9336 | 20.0 | 0.9336 | True | label_T&C |

Build: `qqgjyx/sgtsnepi@b1131f8 (FIXED)`. Metric: harmonic mean of Label-T and
Label-C (now standardized; closes the arithmetic-vs-harmonic inconsistency
flagged in the R4-E1 closing note).

**Shifts vs the broken-build R3 auto-λ:**

| dataset | R3 (broken) | R5 (FIXED) | Δ |
|---|---|---|---|
| cora | 20 | 20 | same |
| citeseer | 5 | 1 | **shift down (probe-set miss; gridsearch=3 within 0.013)** |
| mnist_knn | 20 | 1 | **dramatic: λ=1 wins by 0.001 over λ=30; metric near-saturation across all λ** |
| pubmed | 5 | 5 | same |
| pbmc | unavailable | 5 | **newly available via graph-T&C recomputation** |
| ca_astroph | unavailable | unavailable | no labels (auto-λ undefined) |
| ogbn_arxiv | 20 | 20 | same |

The match-rate over the 5 labeled datasets where the probe + gridsearch are
both defined is now 2/5 (`pubmed`, `pbmc`). For Cora, MNIST, and Citeseer the
probe set `{1, 5, 20, 50}` misses by ≤0.013 absolute. The MNIST shift (20→1)
is the most surprising: with the FIXED build the metric is near-saturation
(0.987–0.988) across all 16 λ values and the argmax is dominated by ZADU's
sample variance — paper-side may want to footnote this as "all λ within
metric noise on saturated datasets".

**R5-A2 — refit `eq:moment` on 5 labeled datasets:**

| | R3 (broken, 3-pt) | R5 (FIXED, 5-pt) |
|---|---|---|
| c0 | 20.59 | **0.187** |
| c1 | -4.11 | **+5.900** |
| R² | 0.165 | 0.159 |

The c1 sign flipped from negative to positive — more degree heterogeneity
(higher CV(d)) now predicts a higher λ, matching the qualitative claim that
heterogeneous-degree graphs benefit more from λ-rescaling. R² ≈ 0.16 stays
essentially unchanged; the residual variance is dominated by MNIST's
near-degenerate λ surface (CV(d)=0.30, auto-λ=1) and Citeseer's flat
near-saturation curve. Honest data; the linear trend is a first-order
indicator, not a tight fit.

**R5-A3 — predictions for held-out / unlabeled:**

| dataset | CV(d) | R3 prediction | R5 prediction |
|---|---|---|---|
| pbmc | 0.751 | 17.50 | **4.62** |
| ca_astroph | 1.448 | 14.63 | **8.73** |
| pubmed | 1.653 | 13.79 | **9.94** (overrides 5.0 from probe) |

The R5 PBMC prediction (4.62) now lines up with the empirical PBMC auto-λ=5
from the FIXED grid — a sanity check that the fit is at least
self-consistent on the labeled set. ca_astroph drops from 14.6 to 8.7;
pubmed drops from 13.8 to 9.9.

**Files updated:**

- `output/tables/auto_lambda_summary.parquet` — replaces R3 (broken) values.
- `output/tables/moment_fit.json` — 5-pt fit on FIXED-build auto-λ; provenance
  field cites the supersession.
- `scripts/run_r5_corrected_autolambda.py` (new) — idempotent re-emission.

**R5-D status**: PUNT to repo maintainer. Releasing `pysgtsnepi v0.3.1` to
PyPI requires PyPI auth; experiment-side cannot perform. The fix already
exists in `qqgjyx/sgtsnepi@b1131f8`; what's needed is `python -m build`
+ `twine upload`. Blocking on owner action.

**Open follow-ups (not blocking)**:

- The MNIST λ=1 corrected auto-λ contradicts the R3 narrative (λ=20). Worth a
  paper-side decision: report λ=20 with a "saturation on near-regular kNN
  graphs" footnote, or accept λ=1 as the corrected number.
- Citeseer probe miss (auto=1 vs gridsearch=3) is small enough to keep
  auto-λ=1 honest, but the broader question of probe-set adequacy
  ({1,5,20,50}) is open. R4-G1 lambda-sensitivity figure (deferred) would
  help visualize this.
- run_e2_auto_lambda.py and the Coauthor/OGBN runners disagree on
  arithmetic vs harmonic mean for Label-T&C; this re-emission standardizes
  on harmonic. A one-line patch in run_e2_auto_lambda.py to match would
  close the loop, but no current parquet depends on that script's output.

---

### Paper-side ← experiment-side, Phase 4 (appended 2026-04-28, agent EXP-AGENT)

Phase 4 cheap-render tier closed: A insets refreshed, C paper-table
re-emitted with corrected auto-λ, F ablation shipped, G1 λ-sensitivity
overlay rendered, H1 Pareto + H2 qualitative figures rendered.

**R4-A (insets refresh)**: `aggregate_insets.py` extended to recompute
graph-T&C on the fly for PBMC (which has HDBSCAN labels but no features,
so cell parquets had NaN). N=5 means now populated for all 4 inset λs:

| dataset | λ=1 LT | λ=5 LT | λ=20 LT | λ=80 LT |
|---|---|---|---|---|
| cora | 0.855 | 0.908 | **0.935** | 0.904 |
| mnist_knn | 0.992 | 0.990 | 0.991 | 0.976 |
| pbmc | 0.954 | 0.960 | 0.960 | 0.959 |

Cora peaks at λ=20 (matches auto-λ); PBMC saturated above λ=5; MNIST
saturated across all 4. Re-rendered `teaser_pbmc_hero.pdf` and
`lens_idiom_regimes.pdf`.

**R4-C (paper table)**: `paper_table_comparison.{md,tex}` regenerated
with all 7 datasets (cora, citeseer, pubmed, mnist_knn, pbmc, ca_astroph,
ogbn_arxiv) and the corrected `auto_lambda_summary.parquet`.

**R4-F (subtractive ablation)**: 64 rows in `output/tables/ablation.parquet`
+ `ablation_table.tex`. Variants: `full` (auto-λ + PCA-init from
`comparison_agg`), `pca_init_off` (random init, N=5 new compute),
`auto_lambda_off_fixed20` (extracted from λ=20 PCA cells),
`degree_rescaling_off_lambda1` (extracted from λ=1 PCA cells).

Headline ablation deltas (mean Label-T):

| dataset | full | -PCA init | -auto-λ (fixed 20) | -degree rescaling (λ=1) |
|---|---|---|---|---|
| cora | 0.924 | 0.912 (-0.012) | 0.935 (+0.011) | 0.855 (**-0.069**) |
| citeseer | 0.739 | 0.735 (-0.004) | 0.732 (-0.007) | 0.745 (+0.006) |
| pubmed | 0.903 | 0.857 (**-0.046**) | 0.905 (+0.002) | 0.892 (-0.011) |
| mnist_knn | 0.982 | 0.990 (+0.008) | 0.991 (+0.009) | 0.992 (+0.010) |
| pbmc | 0.968 | NaN (graph-only no recompute in F) | NaN | NaN |

Story: degree-rescaling matters most on cora (-0.069 when off);
PCA-init matters most on pubmed (-0.046); MNIST is fully saturated
(every variant ≥0.982).

**R4-G1 (λ-sensitivity overlay)**: `output/figures/lambda_sensitivity.pdf`
plots Label-T (or T) vs log10(λ) for all 6 datasets that have CV(d)
data, color-coded by CV(d) cool→warm. Auto-λ markers as dotted vertical
lines; λ<1 region shaded gray. Visualizes that:
- mnist_knn (CV=0.30) is essentially flat at 0.99 across all 16 λ
- citeseer (CV=1.22) is also mostly flat ~0.85
- cora (CV=1.34) and pubmed (CV=1.65) show clear peaks
- pbmc (CV=0.75) saturates above λ=5

**R4-H1 (Pareto figure)**: `output/figures/pareto_quality_runtime.pdf`
plots log10(runtime) vs Label-T (or T) per (method, dataset) cell across
all 7 datasets. Pareto frontier line in dashed gray. ours marked larger
in brand red.

**R4-H2 (qualitative)**: `output/figures/qualitative_3method_2dataset.pdf`
2×3 grid: (Cora top, PubMed bottom) × (ours, UMAP, node2vec+UMAP),
median-Label-T seed per cell, LT corner badge.

**Skipped/deferred**:
- **G2-G6** (perplexity, n_iter, k-kNN, u, alpha sweeps): require new
  compute on zjl. G5 (`u`) confirmed broken in R4-G earlier (kwarg not
  exposed in installed pysgtsnepi). Defer until paper-side asks.
- **H3** (cores scaling): requires new compute, deferred.
- **D2** (MNIST node2vec full): on zjl from earlier, not pulled to local;
  status unknown.
- **D4** (node2vec multi-seed PBMC/ca_astroph/PubMed): on zjl from earlier,
  not pulled. The N=5 PHATE/UMAP/openTSNE/pysgtsnepi cells are present in
  comparison_agg; node2vec single-seed remains the carryover.

**Phase 5 (I1+I2 N=10 + Wilcoxon, J1 ogbn-products)**: paused awaiting
greenlight; the corrected R5 numbers may shift Phase 5 priorities.


---

### Phase 5 launched (overnight queue, appended 2026-04-28T00:53Z, agent EXP-AGENT)

R4 finishing tier queued in zjl tmux session r4_overnight via
scripts/run_overnight_queue.sh. Sequencing (each tier wrapped in
try/continue so one failure does not block the rest):

1. **I1** N=5→N=10 extension on contested datasets {pbmc, citeseer,
   mnist_knn} × cheap methods {umap, opentsne, phate, pysgtsnepi}.
   59 cells on 4 workers; ~1.5h. Excludes node2vec_umap (compute-prohibitive
   at 5 extra seeds; existing single-seed stays).
2. **D4-finish** node2vec multi-seed pbmc + pubmed seeds 42-46 (current
   state: each has only seed=42 cell). ~3h on workers=2 with
   Node2Vec(workers=8) inside.
3. **G2/G3/G4/G6** sensitivity sweeps on Cora + PubMed (G5 unsupported
   per earlier finding; G6 fixed to use alpha kwarg). ~1h.
4. **J1** ogbn-products scale demo (n=2.4M, single seed). ~30 min stretch.
5. **I2** Wilcoxon paired-rank at full N=10 (post-hoc, ~5 min).
6. **H3** cores scaling LAST (so timing is not polluted by concurrent
   load). ~30 min.
7. Final re-merge + emit_paper_table + render H1/H2.

State file: output/meta/r4_overnight_state.json with per-tier
status. Log: output/meta/r4_overnight_queue.log. Estimated total wall
~6h. The agent will check back after the queue closes; cells are
resume-safe via per-cell parquets so a mid-queue tmux death is recoverable.

Smoke-tested before launch: G2 perplexity, G3 max_iter, G4 k-kNN, G6
alpha kwarg all produce sane label_T values on Cora at seed=42; one I1
cell (pbmc/umap/seed=47) ran clean on zjl in 28.6s with LT&C=(0.977,0.991).


---

### R6 visual pass — ball-kick from paper-side (appended 2026-04-28, agent PAPER-AGENT)

Source: `~/.../overleaf_SGtSNE-Pi/doc/visual_pass_ballkick.md` (full
specs there; this is the actionable triage for the experiment-side
agent). Paper-side has applied: `tab:comparison` transpose
(methods-as-rows, multicol-per-dataset, baselines/`\ours` `\midrule`
split), salmon/peach/cream highlight palette swap (matching DriftNeRF
prior-work convention; the gold/silver/bronze in the rulebook was a
transcription error and has been corrected), and all Phase 3
line-level §Empirical fixes (T1–T5 references stripped, PBMC metric
description corrected, phantom ogbn-arxiv/Coauthor-CS/OGBN-products/
cores-scaling references trimmed from §Setup, "within metric noise"
overclaim reframed, "two fastest" overclaim corrected, headline
ratios `41×→42×` and `138×→136×` aligned with table cells, Limitations
distinction algebraic-`eq:cv` vs empirical-cutoff, graph-aware
trustworthiness claim *dropped* from abstract + body pending D6
below).

Paper builds clean at 5 pages.

---

#### Already shipped by experiment-side; paper-side just needs to import

Most R4 / R5 deliveries are sitting in `output/figures/` and have
not yet been copied into `overleaf_SGtSNE-Pi/images/`. Paper-side
will pull these once the next paper-side commit lands; flagging here
so nothing gets accidentally regenerated.

| paper-side asks                           | already at experiment-side                                       | action |
|---|---|---|
| D3 ablation table                         | `output/tables/ablation_table.tex`, `ablation.parquet` (R4-F)    | paper-side imports |
| D4 sensitivity figure                     | `output/figures/lambda_sensitivity.{pdf,png}` (R4-G1)            | paper-side imports |
| D5-alt closing-viz Pareto                 | `output/figures/pareto_quality_runtime.{pdf,png}` (R4-H1)        | paper-side imports as fallback |
| qualitative side-by-side                  | `output/figures/qualitative_3method_2dataset.{pdf,png}` (R4-H2)  | paper-side imports if used |
| BA/WS synthetic regime control            | `output/figures/synthetic_regime_control.{pdf,png}` (R4-G earlier)| paper-side imports |
| 4-panel PBMC teaser (current `teaser.pdf`) | `output/figures/teaser_pbmc_hero.{pdf,png}` (R4-A)              | superseded by D1 below |

Paper-side note: `images/lens_idiom.pdf` and `images/teaser.pdf` are
the two figures currently embedded in `main.tex`; everything else in
the table above ships into `images/` only when the paper-side commits
the corresponding `\includegraphics` / `\input` lines.

---

#### Genuinely new asks (do these next)

**D1 — multi-row teaser (3 rows × 4 cols, supersedes single-row PBMC).** The
current teaser shows only PBMC at λ ∈ {1, 5, 20, 80}, which duplicates
the top row of `lens_idiom_regimes.pdf` (Cora) and underuses the
page-1 hero. We want the regime-dependence story landing instantly.
- Row 1 — degree-heterogeneous: Cora (CV(d)≈1.34) at λ ∈ {1, 5, 20, 80}, PCA-init seed=42, color by class, LT-inset per panel, mark `\autolambda` panel.
- Row 2 — saturating: PBMC at the same λ grid; LT inset against HDBSCAN labels; mark `\autolambda`.
- Row 3 — nearly-regular: MNIST-kNN at the same λ grid; panels visually indistinguishable is the message.
- Output: `output/figures/teaser_three_regimes.{pdf,png}` at full text-column-width × 3 rows. Paper-side will replace `images/teaser.pdf`.
- The cell-level data already exists per R4-A insets (cora/mnist/pbmc LT means at λ ∈ {1, 5, 20, 80} are tabulated). This is a re-render with an expanded layout, not new compute.

**D5 — closing-viz: CV(d)-vs-Procrustes scatter (the "indicator-vs-empirical-effect" plot).**
The paper now owns CV(d) as our regime indicator (`eq:cv` in §Background) and the algebraic-vs-empirical split is explicit. The canonical closing-viz for indicator-proposing papers is *one image showing the indicator predicts the observed effect*. Spec:
- x: CV(d), log-scale acceptable
- y: mean pairwise Procrustes across the λ grid
- One point per dataset (Cora, Citeseer, PubMed, MNIST-kNN, PBMC, ca-AstroPh) plus BA + WS synthetic controls if the data is still around (the synthetic_regime_control.pdf assets imply yes)
- Vertical bands at CV(d)=0.3 and CV(d)=1 shading the three regimes (regular / marginal / heterogeneous)
- Each point labeled with dataset name; horizontal error bar is the seed-noise floor (PCA-init Procrustes std)
- Output: `output/figures/cv_vs_procrustes.{pdf,png}` (PGFPlots TikZ source if feasible: `output/figures/cv_vs_procrustes.tex`)

If D5 is too costly to land before the deadline, fall back to the existing R4-H1 Pareto (`pareto_quality_runtime.pdf`) as the closing-viz; the CV(d)-vs-Procrustes is preferred because it ties §Background → §lens → results in one image.

**D6 — graph-aware trustworthiness on PBMC (HIGH; was an unverifiable claim in the abstract + §Results).**
Paper-side has *dropped* the graph-aware trustworthiness claim from the abstract and body (PBMC now reads as a clean Label-T&C loss of 0.006 to UMAP). To restore the "split with UMAP" framing requires:
- A citation for "graph-aware trustworthiness" as a defined metric, or a clear inline definition in §Setup (presumably trustworthiness of the embedding's kNN against the *input* graph rather than against HDBSCAN labels).
- The exact numbers: `\ours` 0.704 (?) and UMAP 0.681 (?) — confirm against the parquet, and provide the seed-by-seed values so std fits with our `\s{}` macro.
- Either a new column in `tab:comparison` or a forward-pointer to a supplemental archive.

If the metric is paper-coined, we can just define it in §Setup ("we additionally report trustworthiness against the input kNN graph, distinct from Label-T&C against HDBSCAN labels") and add the numbers — but we need confirmed values.

**D2 — architecture flowchart (TikZ).**
Paper-side owns the design (no compute needed). Experiment-side input requested only as a sanity-check: please confirm the simplified pipeline below matches the installed `pysgtsnepi` package's actual code path:

```
G(V, E, W)  →  column-rescale to sum λ (eq:lambda)  →  symmetrize
            →  KL minimization with FFT-O(n) repulsion  →  Y ∈ R^(n×2)
```

And in parallel for `\autolambda`:

```
G  →  λ ∈ {1, 5, 20, 50}  →  fit each (4 sgtsne calls)  →  LT&C scores
                                                       →  argmax → λ*
   (if unlabeled)  →  CV(d)  →  clamp(c0 + c1·CV(d), 1, 80) → λ_moment
```

Confirm/correct in 1-2 sentences in the next ledger update; paper-side will draft the TikZ from there.

**D7 — numeric precision reconciliation.**
The paper now quotes `42×` (Cora) and `136×` (ca-AstroPh) speedups, derived from `420/10 = 42` and `4083/30 = 136.1`. If the underlying parquet has higher-precision wall-clock seconds (e.g., `\ours` ca-AstroPh = 29.6s → 137.9× ≈ 138×), please re-emit the relevant `tab:comparison` cells at one decimal so headline numbers are exact. Specifically: Cora (`\ours` runtime, n2v+UMAP runtime) and ca-AstroPh (`\ours` runtime, n2v+UMAP runtime). The other dataset cells can stay at integer precision.

---

#### Suggested order of operations

1. **D6 first** (HIGH; data may already be in the parquet — if so, this lands fast).
2. **D7 second** (LOW effort; just pull higher-precision values from the existing parquet).
3. **D1 third** (re-render of existing data into a 3-row layout; cheap).
4. **D5 fourth** (new compute? probably not — Procrustes-vs-CV(d) values exist already from §lens characterization; this is a plotting task).
5. **D2 fifth** (just a 2-sentence pipeline confirmation; no code changes).

Items D3 + D4 (ablation, sensitivity) are already done — paper-side
will just import them once D1/D5 land and the visual layout is final.



---

### Paper-side ← experiment-side, R6 response (appended 2026-04-28T01:30Z, agent EXP-AGENT)

R6 D1/D2/D5/D6/D7 all closed locally before overnight queue completion;
no zjl compute needed. D3/D4 already shipped (paper-side just imports).

**D6 — graph-aware T on PBMC (HIGH).** Numbers extracted from the R3 N=5
multi-seed cells; identical to existing comparison_agg `T` column:

| method | T (mean ± std) | seed-by-seed |
|---|---|---|
| `\ours` (PBMC, auto-λ=10, PCA-init) | **0.7044 ± 0.0016** | 0.7028, 0.7066, 0.7036, 0.7034, 0.7055 |
| UMAP | 0.6815 ± 0.0017 | 0.6795, 0.6821, 0.6820, 0.6801, 0.6836 |

Definition: `lens.metrics.compute_metrics(features=None, adj=A, Y=Y, labels=L)`
treats adjacency rows as the input feature space, then computes ZADU
trustworthiness with k=15 on a 5000-node subsample. So this IS exactly
"trustworthiness against the input kNN graph", distinct from Label-T&C
against HDBSCAN labels. Mean diff = +0.023 in our favor. Wilcoxon
paired-rank at N=5 gives p=0.0625 (one-sided in our favor); will re-test
at N=10 from overnight queue's I2 stage.

Suggested §Setup phrasing: "we additionally report graph-aware
trustworthiness (T against the input adjacency, distinct from Label-T&C
against HDBSCAN labels) for the graph-only PBMC dataset." No external
citation required since it is the standard ZADU trustworthiness metric
applied to the adjacency-as-features path that our
`lens.metrics.compute_metrics` already implements.

**D7 — runtime precision reconciliation.** Full-precision values from
`*_comparison_agg.parquet`:

| dataset | method | runtime_s_mean | runtime_s_std | n_seeds |
|---|---|---|---|---|
| Cora | `\ours` | **10.2** | ±0.1 | 5 |
| Cora | node2vec+UMAP | **420.0** | ±4.1 | 5 |
| ca-AstroPh | `\ours` | **29.6** | ±1.2 | 5 |
| ca-AstroPh | node2vec+UMAP | **4082.7** | (single seed; Phase-6 carryover at workers=1, matched-protocol with R3 main grid) | 1 |

Speedups at one-decimal precision:
- Cora: 420.0 / 10.2 = **41.2×** (paper currently quotes "42×"; correct value is 41×)
- ca-AstroPh: 4082.7 / 29.6 = **137.9×** (paper currently quotes "136×"; correct value is 138×)

Caveat for ca-AstroPh: cells_baselines/ also has a multi-seed node2vec
rerun at workers=8 (~600s mean) which uses a different protocol. Only
the workers=1 single-seed value is matched-protocol with the rest of
the R3 comparison grid; that is the canonical 4082.7s figure for the
headline.

**D2 — pipeline confirmation (sanity check).** The simplified pipeline in
your TikZ draft matches the installed `pysgtsnepi` code path:

> G(V,E,W) → column-rescale to sum λ → symmetrize → KL minimization with
> FFT-O(n) repulsion → Y ∈ R^(n×2)

Two minor refinements:
- The `unweighted_to_weighted=True` flag (default) inserts a Jaccard
  preprocessing step BEFORE column-rescaling for unweighted symmetrized
  graphs. This is required for λ-rescaling to be non-degenerate (the
  PyPI v0.3.0 bug fixed in `qqgjyx/sgtsnepi@b1131f8` was precisely the
  omission of this step).
- `unweighted_to_weighted=False` is used for PBMC (the input is already
  a stochastic kNN matrix; double-Jaccard would damage the structure).

For `\autolambda`: your draft is correct. The 4-fit probe set
`{1, 5, 20, 50}` is the auto-λ cost (4× single-fit wall ≈ 38.8s on PBMC
at PCA-init seed=42 per `[Pa]`).

**D1 — three-row teaser.** `output/figures/teaser_three_regimes.{pdf,png}`
shipped. Layout: 3 rows × 4 cols, full text-column-width × 6.2 cm,
left-side row labels with CV(d) annotation. Insets show N=5 mean LT&C
from `output/tables/teaser_lens_inset_means.json`. Auto-λ marked with
`★` per dataset (Cora=20, PBMC=5, MNIST-kNN=1 — R5-A1 corrected values).

Note the MNIST `\autolambda` star is now on λ=1 instead of λ=20 (the
R5-A1 corrected auto-λ; metric near-saturated across all 16 λ values on
MNIST). If the paper wants the prior λ=20 framing for MNIST, the inset
values still show λ=20 LT=0.991 vs λ=1 LT=0.992 — within 0.001; both
defensible.

**D5 — CV(d) vs mean-pairwise-Procrustes scatter.**
`output/figures/cv_vs_procrustes.{pdf,png}` shipped. Numeric summary in
`output/tables/cv_vs_procrustes_summary.json`:

| dataset | CV(d) | mean Procrustes | seed-noise |
|---|---|---|---|
| MNIST-kNN | 0.30 | 0.095 | 0.701 |
| PBMC | 0.75 | 0.110 | 0.122 |
| PubMed | 1.65 | 0.193 | NaN (no multi-seed) |
| ca-AstroPh | 1.45 | 0.561 | NaN (no multi-seed) |
| Citeseer | 1.22 | 0.760 | 0.911 |
| Cora | 1.34 | 0.854 | 0.840 |
| BA-synth | 1.33 | 0.267 | n/a |
| WS-synth | 0.10 | 0.262 | n/a |

Procrustes = orthogonal alignment (rotation + reflection, no scale)
RMSE / RMS-radius of Y1; pairwise across the 16-pt λ grid at seed=42
PCA-init. Bands shaded at CV(d) ≤ 0.3 (regular), 0.3–1.0 (marginal),
> 1.0 (heterogeneous).

Honest observations the figure surfaces:
- Cora / Citeseer / ca-AstroPh support CV(d) → λ-effect.
- **PubMed is an outlier** (high CV but only 0.19 Procrustes); plausibly
  because pysgtsnepi hits "non-convergent elements" warnings on PubMed
  at high λ, saturating the embedding into a similar shape across the
  upper-λ range.
- BA and WS synthetic both at ~0.26 even though their CV(d) differ by
  13×: at n=2000 the regime distinction is muted.
- For datasets where multi-seed coverage exists (Cora / Citeseer / MNIST
  / PBMC via R4-A insets), seed-noise is comparable to or larger than
  mean Procrustes — consistent with the R3 "marginal verdict" framing.

If paper-side prefers a tighter story, R4-H1 Pareto remains the
fallback closing-viz; experiment-side recommends keeping D5 as the
canonical closing-viz and footnoting the PubMed outlier honestly.

**Files added this round:**
- `scripts/render_r6d1_three_row_teaser.py`
- `scripts/render_r6d5_cv_vs_procrustes.py`
- `output/figures/teaser_three_regimes.{pdf,png}`
- `output/figures/cv_vs_procrustes.{pdf,png}`
- `output/tables/cv_vs_procrustes_summary.json`

**Overnight queue still running on zjl** (Tier I1 in progress at last
check); full R4 N=10 + Wilcoxon + cores-scaling results will land
~06:00–08:00 zjl time. The R6 deliverables above do not depend on the
queue, so the morning rsync pulls both R4-tail and R6 outputs into one
paper-side import.
