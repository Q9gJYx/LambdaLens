# sgtsnepi

Experimental collaboration repository for preparing the VIS 2026 short paper for `qqgjyx/sgtsnepi`.

## Repository structure

```text
.
├── data/
│   ├── external/   # third-party source data
│   ├── interim/    # intermediate transformation outputs
│   ├── processed/  # cleaned/final datasets used in analysis
│   └── raw/        # immutable raw snapshots
├── docs/           # notes and paper-related documentation
├── notebooks/      # exploratory notebooks
├── output/
│   ├── figures/    # generated figures for manuscript/slides
│   └── tables/     # generated tables/results
├── src/            # reusable project code (if needed later)
└── tests/          # tests (if code is added later)
```

## Environment management (uv)

1. Install [uv](https://docs.astral.sh/uv/).
2. Create/sync environment:

   ```bash
   uv sync
   ```

3. Run tools in the environment:

   ```bash
   uv run pytest
   ```

## Collaboration notes

- Keep `data/raw/` immutable once data snapshots are added.
- Do not commit large generated outputs; only keep reproducible artifacts or lightweight examples.
- Add paper workflow docs in `docs/` and exploratory analysis in `notebooks/`.
