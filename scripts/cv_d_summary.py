"""P2-13: per-dataset CV(d) summary. Used by [Pb] degree-moment fit and §2 of paper."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from lens.data import load_dataset

DATASETS = ("cora", "citeseer", "pubmed", "mnist_knn", "ca_astroph", "pbmc")


def main() -> int:
    out: dict[str, dict[str, float | int]] = {}
    for ds in DATASETS:
        try:
            adj, _, _ = load_dataset(ds)
        except Exception as e:
            print(f"[cv_d] {ds}: SKIP ({type(e).__name__}: {e})", flush=True)
            continue
        d = np.asarray(adj.sum(axis=1)).ravel()
        nz = d[d > 0]
        out[ds] = {
            "n": int(adj.shape[0]),
            "n_nonisolated": int(nz.size),
            "mean_degree": float(nz.mean()),
            "std_degree": float(nz.std()),
            "cv_d": float(nz.std() / nz.mean()) if nz.mean() > 0 else float("nan"),
            "max_degree": int(nz.max()),
        }
        print(f"[cv_d] {ds}: n={out[ds]['n']} mean={out[ds]['mean_degree']:.2f} "
              f"std={out[ds]['std_degree']:.2f} CV={out[ds]['cv_d']:.3f}", flush=True)
    Path("output/tables").mkdir(parents=True, exist_ok=True)
    with open("output/tables/cv_d_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cv_d] wrote output/tables/cv_d_summary.json ({len(out)} datasets)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
