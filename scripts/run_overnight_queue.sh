#!/usr/bin/env bash
# R4 overnight queue: chain remaining experiments sequentially with hard
# per-tier wall-clock caps so a single stuck worker can't eat the night.
#
# Resume-safe via per-cell parquets; relaunching skips done cells.
# Each tier:
#   - hard timeout (kills the tier when exceeded; queue continues)
#   - state mark per outcome: completed / timeout / failed
#   - log appended to a single overnight log
#
# Sequencing rationale:
#   I1 -> D4 -> G2/G3/G4/G6 -> J1 -> I2 -> H3 last (clean box for timing)
#   Final: re-merge cells, re-emit paper table, re-render H1/H2.

set -u
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"

cd "${REPO_ROOT:-$HOME/WorkSpace/wh/SGtSNE-Pi}"
LOG_DIR="output/meta"
mkdir -p "$LOG_DIR"
QUEUE_LOG="$LOG_DIR/r4_overnight_queue.log"
STATE_JSON="$LOG_DIR/r4_overnight_state.json"

date_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
banner() { printf "\n========== [%s] %s ==========\n" "$(date_utc)" "$1" | tee -a "$QUEUE_LOG"; }

# Atomic state.json updater. Writes to a tempfile then renames; tolerates
# malformed prior content.
mark() {
    local key="$1"
    local val="$2"
    python3 - "$STATE_JSON" "$key" "$val" "$(date_utc)" <<'PYEOF'
import json, os, sys, tempfile
path, key, val, ts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = json.load(open(path)) if os.path.exists(path) else {}
except Exception:
    d = {}
d[key] = val
d["updated"] = ts
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
with os.fdopen(fd, "w") as f:
    json.dump(d, f, indent=2)
os.replace(tmp, path)
PYEOF
}

# Run a tier with a hard timeout. $1 = state-key, $2 = wall-cap, $3..$N = cmd.
# Note: this script intentionally does NOT use `set -e`; failures of any
# tier should be marked and the queue must continue. `mark` calls are
# defensive (`|| true`) so a transient JSON-write failure can't kill the
# whole queue.
run_tier() {
    local key="$1"; shift
    local cap="$1"; shift
    mark "$key" "running" || true
    banner "Tier $key (cap=$cap): $*"
    timeout --kill-after=30s "$cap" "$@" >> "$QUEUE_LOG" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        mark "$key" "completed" || true
        banner "Tier $key: COMPLETE (rc=0)"
    elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        mark "$key" "timeout" || true
        banner "Tier $key: TIMEOUT after $cap (rc=$rc); continuing"
    else
        mark "$key" "failed" || true
        banner "Tier $key: FAILED (rc=$rc); continuing"
    fi
    return 0
}

# Pre-flight: disk + tmux + python sanity
banner "Pre-flight"
free_gb=$(df -BG "$PWD" | awk 'NR==2 {print $4}' | tr -d 'G')
echo "  disk free: ${free_gb}G" | tee -a "$QUEUE_LOG"
if ! [[ "${free_gb:-}" =~ ^[0-9]+$ ]] || [ "${free_gb:-0}" -lt 30 ]; then
    banner "ABORT: <30G free disk (got '${free_gb:-?}'); queue not safe to launch"
    mark "queue_status" "aborted_disk" || true
    exit 1
fi
uv run python -c "import lens.data, lens.init, lens.metrics, lens.run; print('  lens import ok')" >> "$QUEUE_LOG" 2>&1 || {
    banner "ABORT: lens package import failed"
    mark "queue_status" "aborted_import" || true
    exit 1
}
echo "  python: $(uv run python --version 2>&1)" | tee -a "$QUEUE_LOG"
mark "queue_status" "running" || true
mark "started" "$(date_utc)" || true

# === Tier I1: N=5 -> N=10 extension on contested datasets, cheap methods ===
run_tier i1 2.5h \
    uv run python scripts/run_e3_baselines.py \
    --datasets pbmc citeseer mnist_knn \
    --methods pysgtsnepi umap opentsne phate \
    --seeds 47 48 49 50 51 \
    --max-workers 4 \
    --phate-subsample-n 100000

# === Tier D4-finish: node2vec multi-seed pbmc + pubmed (have seed=42) ===
# 4 cells × ~1600s on pbmc + 4 × ~1100s on pubmed = ~3h, cap at 4h
run_tier d4 4h \
    uv run python scripts/run_r4d_node2vec_multiseed.py \
    --datasets pbmc pubmed \
    --seeds 42 43 44 45 46 \
    --max-workers 2

# === Tier G2/G3/G4/G6 sensitivity sweeps ===
run_tier g 1.5h \
    uv run python scripts/run_r4g_sensitivity.py \
    --sweeps g2 g3 g4 g6 \
    --datasets cora pubmed \
    --max-workers 4

# === Tier J1: ogbn-products scale demo (n=2.4M) ===
run_tier j1 2h \
    uv run python scripts/run_r4j1_ogbn_products.py

# === Tier I2: Wilcoxon (post-hoc on N=10 results) ===
run_tier i2 10m \
    uv run python scripts/run_r4i2_wilcoxon.py

# === Tier H3: cores scaling LAST so timing isn't polluted ===
run_tier h3 1.5h \
    uv run python scripts/render_r4h_figures.py \
    --figures h3 --max-workers 1,2,4,8,16,32

# === Final: re-merge, re-emit, re-render (set +e not needed; never on globally) ===
banner "Final: re-merge, emit table, render H1/H2"
timeout 10m uv run python scripts/merge_e3_results.py >> "$QUEUE_LOG" 2>&1 || banner "  merge_e3_results: rc=$?"
timeout 5m  uv run python scripts/emit_paper_table.py  >> "$QUEUE_LOG" 2>&1 || banner "  emit_paper_table: rc=$?"
timeout 10m uv run python scripts/render_r4h_figures.py --figures h1 h2 >> "$QUEUE_LOG" 2>&1 || banner "  render H1/H2: rc=$?"

mark "queue_status" "completed" || true
mark "ended" "$(date_utc)" || true
banner "OVERNIGHT QUEUE COMPLETE"
echo "Final state:" | tee -a "$QUEUE_LOG"
cat "$STATE_JSON" | tee -a "$QUEUE_LOG"

# === Post-queue summary written to a separate file for easy parsing ===
SUMMARY="$LOG_DIR/r4_overnight_summary.txt"
{
    echo "R4 overnight queue summary  $(date_utc)"
    echo "==========================================="
    echo "State JSON:"
    cat "$STATE_JSON"
    echo
    echo "Cells in cells_baselines:"
    ls output/tables/cells_baselines/ 2>/dev/null | wc -l
    echo
    echo "Per (dataset, method) cell counts:"
    for ds in pbmc citeseer mnist_knn cora pubmed ca_astroph ogbn_arxiv; do
        for m in pysgtsnepi umap opentsne phate node2vec_umap; do
            c=$(find output/tables/cells_baselines -maxdepth 2 -name "${ds}_${m}_seed*.parquet" 2>/dev/null | wc -l)
            printf "  %-12s/%-15s: %d\n" "$ds" "$m" "$c"
        done
    done
    echo
    echo "Last 30 log lines:"
    tail -30 "$QUEUE_LOG"
} > "$SUMMARY"

banner "Summary written to $SUMMARY"
