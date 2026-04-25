# CLAUDE.md

Guidance for Claude Code working on the experiment-side of the
**Lambda Lens** IEEE VIS 2026 short paper.

## What this repo is

Experiment companion repo: the `lens` analysis package, notebooks,
scripts, and the artifact tree under
`output/{tables,embeddings,figures,meta}/`. The paper repo
(`overleaf_SGtSNE-Pi`) reads from here; we never write back to it.

## Three-repo split — do not confuse

| Repo | Role | This agent writes? |
|---|---|---|
| `_Paper/overleaf_SGtSNE-Pi/` | LaTeX paper | **No.** Read-only. |
| `_Projects/SGtSNE-Pi/` (this) | Experiments + artifacts | **Yes.** Default target. |
| `_Projects/sgtsnepi/` | PyPI package `pysgtsnepi` | **No — keep clean.** Consume as dep only. |

## Canonical sources of truth (read, do not duplicate)

- Mission / 5-day plan: `@_Paper/overleaf_SGtSNE-Pi/doc/outline.md` and `@_Paper/overleaf_SGtSNE-Pi/doc/experiment_plan.md`.
- Two-repo coordination protocol: `@_Paper/overleaf_SGtSNE-Pi/doc/coordination.md`.
- Operating manual for this agent: `@_Paper/overleaf_SGtSNE-Pi/doc/code_repo_init_prompt.md`.
- Status surface (paper-side reads): `PAPER_VIS2026.md` here.
- Machine state: `state.json` here.

If a paper-side change is needed, append a one-line entry under
"Requests for paper-side agent" in `PAPER_VIS2026.md` — never edit the
paper repo directly.

## Branch convention

Work on `paper-vis2026`. `main` stays clean.

## Compute (deviates from manual — Duke CS lost 2026-04-25)

Primary host: **zjl** (`shimarin.vul337.team`, 32c/64t Xeon Gold 6326,
377 GB RAM, no scheduler). Coordinate with zhoujunlin; use the reserved
slot at `/home/zhoujunlin/WorkSpace/wh/lambda-lens`. The (dataset, λ,
seed) grid runs through `scripts/run_e1_local.py` driven by a
`concurrent.futures.ProcessPoolExecutor` (the manual's sbatch template
is moot). Per-experiment state lives in `state.json` under
`runs.{e1,e2,e3,e5_surrogate}`. Backup hosts: xulab (10c/20t / 125 GB),
local Mac (sanity only).

## uv commands

```
uv sync                                    # install/update env
uv run pytest -q                           # smoke tests
uv run python scripts/run_e1.py --dataset cora --lambda 1.0 --seed 42
bash init.sh                               # bootstrap a fresh session
```

## Do-not list

- Don't write to `_Paper/overleaf_SGtSNE-Pi/` or `_Projects/sgtsnepi/`.
- Don't kick off long remote jobs without explicit user approval (manual's autonomy block).
- Don't commit anything under `data/raw/`, `data/processed/`, or `output/embeddings/` (gitignored; reproducible).
