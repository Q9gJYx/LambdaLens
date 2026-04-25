#!/usr/bin/env bash
set -euo pipefail
echo "== environment =="
uv sync
echo "== state.json =="
cat state.json
echo "== PAPER_VIS2026.md =="
cat PAPER_VIS2026.md
echo "== git status =="
git status -s
echo "== last 5 commits =="
git log --oneline -5
