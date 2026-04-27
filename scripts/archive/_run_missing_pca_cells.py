"""One-off: fill λ ∈ {1, 5, 80} PCA-init seed=42 cells for cora + mnist_knn for fig:lens."""
from __future__ import annotations
import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _w(ds: str, lam: float) -> dict:
    from lens.data import load_dataset
    from lens.run import run_one_cell
    adj, features, labels = load_dataset(ds)
    return run_one_cell(adj, features, labels, lam, 42, ds, "output", init="pca")


def main() -> int:
    cells = [(ds, lam) for ds in ("cora", "mnist_knn") for lam in (1.0, 5.0, 80.0)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_w, ds, lam): (ds, lam) for ds, lam in cells}
        for f in as_completed(futs):
            ds, lam = futs[f]
            try:
                r = f.result()
                print(f"OK {ds} lam={lam} {r['runtime_s']:.1f}s", flush=True)
            except Exception as e:
                print(f"FAIL {ds} lam={lam} {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
