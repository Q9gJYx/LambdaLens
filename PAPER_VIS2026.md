# VIS 2026 Paper Status (Lambda Lens)

> Last updated: 2026-04-26 by experiment-side agent

## Headline
- Day: 2 of 5 (Sun 2026-04-26, today). Day 1 work (E1 + kill switch) shifted to today after compute target had to pivot three times (Duke CS → zjl → h17 → Azure).
- Kill switch: ARMED (decision still pending E1 results)
- Blocker: none — VM up, Phase 3 sanity in progress

## Compute
- Host: **Azure VM `lens-vm`** in resource group `vis2026-jp`, region **`japaneast`** (Azure-for-Students subscription is region-restricted to 5 Asia regions; eastus blocked by `sys.regionrestriction` policy). Public IP `40.115.138.51`. SSH alias `lens-vm`.
- Size: **`Standard_E4s_v3`** — 4 vCPU, 32 GB RAM, no GPU (~$0.252/h on-demand).
- Install command: `uv sync` (after rsync)
- Runner: `scripts/run_e1_local.py` driven by `concurrent.futures.ProcessPoolExecutor`, max-workers 2 (4 vCPU / 2 cores per task)
- Cost ceiling: ~$25 of the $691 student credit; `az vm deallocate -g vis2026-jp -n lens-vm` between work bursts

## Experiments

### E1 — lambda grid effect (Day 1 work, running Day 2)
- Status: not started
- Job: n/a
- Recovery: `ssh azure-vm 'pgrep -af run_e1_local.py'` (alias TBD)
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

- **Compute pivot, second iteration.** zjl (manual's intended fallback after Duke CS) turned out to be on a tailnet not shared with this account; h17 has disk pressure (root 99%). Settled on Azure VM (`Standard_*` in eastus) on the user's $691 Azure-for-Students credit. The `code_repo_init_prompt.md` SLURM template (l. 315–341) and Phase 4 wording are doubly moot for this cycle.
- **Plan date alignment.** Original `outline.md` plan started "Day 1 (Sat)" on the 26th but Sat was the 25th. Schedule slipped one day due to the compute search; today (2026-04-26) is doing Day 1 + Day 2 work. AoE deadline 2026-04-30 still feasible if E1 + E2 land today.
- **NN surrogate research add-on.** User asked us to attempt a λ-conditioned GNN+FiLM surrogate (E5) this cycle. Best-effort, not guaranteed. If it lands, may motivate a one-paragraph mention in §5 / Future Work.
