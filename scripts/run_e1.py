"""CLI: one (dataset, lambda, seed) tuple -> parquet row + embedding."""
from __future__ import annotations

import argparse

from lens.data import load_planetoid
from lens.run import run_one_cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cora", "citeseer"])
    p.add_argument("--lambda", dest="lambda_", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", default="output")
    args = p.parse_args()

    adj, features, labels = load_planetoid(args.dataset)
    row = run_one_cell(
        adj, features, labels, args.lambda_, args.seed, args.dataset, args.out_root
    )
    print(row)


if __name__ == "__main__":
    main()
