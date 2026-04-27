"""P1-5: PubMed E1 (small lambda grid) + E2 (auto-lambda probe).

Runs PubMed cells at lambda in {1, 2, 5, 10, 20, 50, 80}, seed=42, PCA-init,
in parallel. Then re-runs the auto-lambda summary script across all 6
datasets (cora, citeseer, mnist_knn, ca_astroph, pbmc, pubmed) to refresh
output/tables/auto_lambda_summary.parquet.

Pubmed adds ~5 cells * ~30-60s each = 3-5 min wall on 8 workers.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

LAMBDAS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0)
SEED = 42


def _w(lam: float) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell
    adj, features, labels = load_dataset("pubmed")
    return run_one_cell(adj, features, labels, lam, SEED, "pubmed", "output", init="pca")


def main() -> int:
    print(f"[pubmed-e1] {len(LAMBDAS)} cells: lambdas={LAMBDAS}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = {ex.submit(_w, lam): lam for lam in LAMBDAS}
        for f in as_completed(futs):
            lam = futs[f]
            try:
                r = f.result()
                print(f"[pubmed-e1] lam={lam} {r['runtime_s']:.1f}s "
                      f"label_T={r['label_trustworthiness']:.3f} "
                      f"label_C={r['label_continuity']:.3f}", flush=True)
            except Exception as e:
                print(f"[pubmed-e1] FAIL lam={lam}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
    print(f"[pubmed-e1] {time.perf_counter() - t0:.1f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
