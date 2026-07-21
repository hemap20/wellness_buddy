#!/usr/bin/env bash
# Commits any pending changes every N minutes, regardless of what the
# training/testing pipeline is doing. Simpler and less precise than
# event-driven commits (e.g. "commit after each phase finishes") — this just
# guarantees you never lose more than N minutes of results/config changes,
# without the pipeline code needing to know anything about git.
#
# Usage:
#   ./auto_commit.sh [interval_minutes]   # default 15
#   nohup ./auto_commit.sh 10 > logs/auto_commit.log 2>&1 &
#
# Stop it with: pkill -f auto_commit.sh

set -euo pipefail
cd "$(dirname "$0")"

INTERVAL_MINUTES="${1:-15}"
INTERVAL_SECONDS=$((INTERVAL_MINUTES * 60))

echo "[auto_commit] starting, committing every ${INTERVAL_MINUTES} min (pid $$)"

while true; do
    sleep "$INTERVAL_SECONDS"
    if [ -n "$(git status --porcelain)" ]; then
        git add -A
        git commit -q -m "auto: periodic snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "[auto_commit] committed at $(date)"
    else
        echo "[auto_commit] nothing to commit at $(date)"
    fi
done
