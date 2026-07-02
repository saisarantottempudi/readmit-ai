#!/usr/bin/env bash
# Bootstrap then monitor loop: ingest + first training if no data yet,
# then run the drift monitor every MONITOR_INTERVAL seconds.
set -euo pipefail

INTERVAL="${MONITOR_INTERVAL:-300}"

if [ ! -f /data/processed/train.parquet ]; then
    echo "[jobs] no processed data found — running ingest flow"
    python -m flows.ingest_flow
fi

echo "[jobs] running training flow"
python -m flows.train_flow

echo "[jobs] entering monitor loop (every ${INTERVAL}s)"
while true; do
    sleep "$INTERVAL"
    python -m flows.monitor_flow || echo "[jobs] monitor flow failed; retrying next cycle"
done
