#!/usr/bin/env bash
# R4 overnight queue: chain remaining experiments sequentially.
# Run inside a tmux session on zjl after `git pull`. All cells are
# resume-safe via per-cell parquets, so this is idempotent.
#
# Sequencing rationale:
#   1. I1 first (N=10 extension on contested datasets) — cheap methods only,
#      ~1.5h; lowest variance, blocks Wilcoxon.
#   2. D4 finish (node2vec mid-seeds for pbmc/pubmed) — ~2h; uses workers=8.
#   3. Sensitivity sweeps G2/G3/G4/G6 — small cells, ~1h total.
#   4. J1 ogbn-products scale demo — ~30 min stretch.
#   5. H3 cores scaling LAST so its timing readings aren't polluted by
#      concurrent box load.
#
# Each tier runs in foreground inside this script (so we get sequential
# logs) but uses ProcessPool internally for parallelism within the tier.
# A failure in any tier is logged and the queue continues to the next.

set -u
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT:-$HOME/WorkSpace/wh/SGtSNE-Pi}"
LOG_DIR="output/meta"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/r4_overnight_queue.log"
STATE_JSON="$LOG_DIR/r4_overnight_state.json"

date_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { printf "\n========== [%s] %s ==========\n" "$(date_utc)" "$1" | tee -a "$QUEUE_LOG"; }
mark()   { python3 -c "import json,sys; d=json.load(open('$STATE_JSON')) if __import__('os').path.exists('$STATE_JSON') else {}; d['$1']='$2'; d['updated']='$(date_utc)'; json.dump(d, open('$STATE_JSON','w'), indent=2)"; }

# Initialize state
[ ! -f "$STATE_JSON" ] && echo '{"started": "'"$(date_utc)"'"}' > "$STATE_JSON"

# Tier I1: N=5 → N=10 extension on contested datasets, cheap methods
banner "Tier I1: N=10 extension {pbmc,citeseer,mnist_knn} × {umap,opentsne,phate,pysgtsnepi}"
mark "i1_status" "running"
if uv run python scripts/run_e3_baselines.py \
    --datasets pbmc citeseer mnist_knn \
    --methods pysgtsnepi umap opentsne phate \
    --seeds 47 48 49 50 51 \
    --max-workers 4 \
    --phate-subsample-n 100000 \
    >> "$QUEUE_LOG" 2>&1; then
    mark "i1_status" "completed"
    banner "Tier I1: COMPLETE"
else
    mark "i1_status" "failed"
    banner "Tier I1: FAILED (continuing queue)"
fi

# Tier D4 finish: node2vec multi-seed on pbmc + pubmed (have seed=42; need 43-46)
banner "Tier D4-finish: node2vec multi-seed pbmc+pubmed seeds 43-46"
mark "d4_status" "running"
if uv run python scripts/run_r4d_node2vec_multiseed.py \
    --datasets pbmc pubmed \
    --seeds 42 43 44 45 46 \
    --max-workers 2 \
    >> "$QUEUE_LOG" 2>&1; then
    mark "d4_status" "completed"
    banner "Tier D4-finish: COMPLETE"
else
    mark "d4_status" "failed"
    banner "Tier D4-finish: FAILED (continuing)"
fi

# Tier G2 + G3 + G4 + G6 sensitivity sweeps
banner "Tier G2/G3/G4/G6: openTSNE perplexity, pysgtsnepi n_iter, k-kNN, alpha"
mark "g_status" "running"
if uv run python scripts/run_r4g_sensitivity.py \
    --sweeps g2 g3 g4 g6 \
    --datasets cora pubmed \
    --max-workers 4 \
    >> "$QUEUE_LOG" 2>&1; then
    mark "g_status" "completed"
    banner "Tier G: COMPLETE"
else
    mark "g_status" "failed"
    banner "Tier G: FAILED (continuing)"
fi

# Tier J1: ogbn-products scale demo (stretch)
banner "Tier J1: ogbn-products scale demo (n=2.4M, single seed)"
mark "j1_status" "running"
if uv run python -c "
import time, numpy as np
from pathlib import Path
import pandas as pd
print('[J1] loading ogbn-products...')
from ogb.nodeproppred import NodePropPredDataset
import scipy.sparse as sp
ds = NodePropPredDataset(name='ogbn-products', root='data/processed/ogbn_products')
graph, labels = ds[0]
n = int(graph['num_nodes'])
edge_index = graph['edge_index']
adj = sp.csr_matrix((np.ones(edge_index.shape[1], np.float32), (edge_index[0], edge_index[1])), shape=(n, n))
adj = ((adj + adj.T) > 0).astype(np.float32); adj.setdiag(0); adj.eliminate_zeros()
labels = np.asarray(labels, dtype=int).ravel()
print(f'[J1] n={n} m={adj.nnz//2} loaded')
from lens.init import pca_init
from pysgtsnepi import sgtsnepi
print('[J1] computing PCA-init Y0...')
Y0 = pca_init(adj, d=2, scale=1e-4, random_state=42)
print('[J1] running pysgtsnepi auto-lambda=20 seed=42 PCA-init...')
t0 = time.perf_counter()
Y = sgtsnepi(adj, d=2, lambda_=20.0, random_state=42, Y0=Y0)
rt = time.perf_counter() - t0
print(f'[J1] DONE in {rt:.1f}s ({rt/60:.1f} min)')
np.save('output/embeddings_baselines/ogbn_products_pysgtsnepi_seed42.npy', np.asarray(Y, np.float64))
pd.DataFrame([{'dataset':'ogbn_products','n':n,'m':adj.nnz//2,'method':'pysgtsnepi','seed':42,'lambda_':20.0,'runtime_s':rt}]).to_parquet('output/tables/ogbn_products_scale.parquet', index=False)
print('[J1] wrote ogbn_products_scale.parquet')
" >> "$QUEUE_LOG" 2>&1; then
    mark "j1_status" "completed"
    banner "Tier J1: COMPLETE"
else
    mark "j1_status" "failed"
    banner "Tier J1: FAILED (continuing)"
fi

# Tier I2: Wilcoxon (post-hoc on N=10 results)
banner "Tier I2: Wilcoxon paired-rank at N=10"
mark "i2_status" "running"
if uv run python -c "
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
out = []
contested = {'pbmc': 'umap', 'citeseer': 'phate', 'mnist_knn': 'umap'}
for ds, vs_method in contested.items():
    seeds = list(range(42, 52))
    p_ours = [Path(f'output/tables/cells_baselines/{ds}_pysgtsnepi_seed{s}.parquet') for s in seeds]
    p_other = [Path(f'output/tables/cells_baselines/{ds}_{vs_method}_seed{s}.parquet') for s in seeds]
    paired_lts = []
    for po, pt in zip(p_ours, p_other):
        if po.exists() and pt.exists():
            lo = float(pd.read_parquet(po)['label_trustworthiness'].iloc[0])
            lt = float(pd.read_parquet(pt)['label_trustworthiness'].iloc[0])
            if not (np.isnan(lo) or np.isnan(lt)):
                paired_lts.append((lo, lt))
    if len(paired_lts) >= 5:
        ours_lt = np.array([p[0] for p in paired_lts])
        other_lt = np.array([p[1] for p in paired_lts])
        diff = ours_lt - other_lt
        try:
            stat, pvalue = wilcoxon(ours_lt, other_lt)
        except Exception as e:
            stat, pvalue = float('nan'), float('nan')
        out.append({'dataset': ds, 'method_a': 'pysgtsnepi', 'method_b': vs_method,
                    'n_pairs': len(paired_lts), 'mean_diff': float(diff.mean()),
                    'wilcoxon_p': float(pvalue), 'wilcoxon_stat': float(stat)})
        print(f'[I2] {ds}: ours vs {vs_method} N={len(paired_lts)} mean_diff={diff.mean():.4f} p={pvalue:.4f}')
df = pd.DataFrame(out)
df.to_parquet('output/tables/wilcoxon.parquet', index=False)
print(f'[I2] wrote wilcoxon.parquet ({len(df)} rows)')
" >> "$QUEUE_LOG" 2>&1; then
    mark "i2_status" "completed"
    banner "Tier I2: COMPLETE"
else
    mark "i2_status" "failed"
    banner "Tier I2: FAILED"
fi

# Tier H3: cores scaling LAST so timing isn't polluted
banner "Tier H3: pysgtsnepi cores scaling on Cora + ca_astroph"
mark "h3_status" "running"
if uv run python scripts/render_r4h_figures.py \
    --figures h3 \
    --max-workers 1,2,4,8,16,32 \
    >> "$QUEUE_LOG" 2>&1; then
    mark "h3_status" "completed"
    banner "Tier H3: COMPLETE"
else
    mark "h3_status" "failed"
    banner "Tier H3: FAILED"
fi

# Re-aggregate comparison tables and emit paper table
banner "Final: re-merge cells, re-aggregate, re-emit paper table"
uv run python scripts/merge_e3_results.py >> "$QUEUE_LOG" 2>&1 || true
uv run python scripts/emit_paper_table.py >> "$QUEUE_LOG" 2>&1 || true
uv run python scripts/render_r4h_figures.py --figures h1 h2 >> "$QUEUE_LOG" 2>&1 || true

mark "queue_status" "completed"
mark "ended" "$(date_utc)"
banner "OVERNIGHT QUEUE COMPLETE"
echo "Final state:" | tee -a "$QUEUE_LOG"
cat "$STATE_JSON" | tee -a "$QUEUE_LOG"
