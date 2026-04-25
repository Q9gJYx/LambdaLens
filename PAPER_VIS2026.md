# VIS 2026 Paper Status (Lambda Lens)

> Last updated: 2026-04-25 by experiment-side agent

## Headline
- Day: 1 of 5 (Sat 2026-04-25, today)
- Kill switch: ARMED
- Blocker: awaiting user answers to Phase 1 questions before initialising remote env

## Compute
- Host: **zjl** (`shimarin.vul337.team`), 32c/64t Xeon Gold 6326, 377 GB RAM, no scheduler
- Remote path: `/home/zhoujunlin/WorkSpace/wh/lambda-lens`
- Install command: `uv sync`
- Runner: `scripts/run_e1_local.py` driven by `concurrent.futures.ProcessPoolExecutor`, max-workers 16
- Backup: xulab (10c/20t / 125 GB), local Mac (sanity only)

## Experiments

### E1 — lambda grid effect (Day 1)
- Status: not started
- Job: n/a
- Recovery: `ssh zjl 'pgrep -af run_e1_local.py'`
- Procrustes gate: pending
- Artifacts: `output/tables/{cora,citeseer,mnist_knn,ca_astroph}_lambda_grid.parquet`
- Notes: 96-cell grid (4 datasets × 8 lambdas × 3 seeds)

### E2 — auto-lambda heuristic (Day 2)
- Status: not started

### E3 — baseline comparison (Day 3)
- Status: not started

### E4 — ipywidget demo (supplementary)
- Status: not started

### E5 — GNN+FiLM lambda-conditioned surrogate (Day 3-4, research add-on)
- Status: not started
- Goal: take (G, λ) → predict embedding without invoking SG-t-SNE-Π. Out of paper scope as a contribution; intended as a future-work pointer in §5. If it lands cleanly may surface as supplementary; if not, no impact on the headline paper.
- Risk: deadline-tight; first to be cut.

## Requests for paper-side agent

- **Compute pivot.** Duke CS access lost on 2026-04-25. Switched to zjl + `ProcessPoolExecutor` instead of SLURM. The `code_repo_init_prompt.md` SLURM template (l. 315–341) and Phase 4 wording need a sync. Functionally equivalent (same artifact tree, same gate semantics).
- **Plan date alignment.** `outline.md` reads "Day 1 (Sat 2026-04-26)" but Sat is **2026-04-25**, today. Numbering shifted accordingly here. Either fix the dates upstream or accept the de-facto shift.
- **NN surrogate research add-on.** User asked us to attempt a λ-conditioned GNN+FiLM surrogate (E5) this cycle. Best-effort, not guaranteed. If it lands, may motivate a one-paragraph mention in §5 / Future Work.
