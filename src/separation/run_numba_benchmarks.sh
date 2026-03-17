#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../../.venv/bin/python}"
DATASET="${DATASET:-$SCRIPT_DIR/../data/galaxy_1000}"
DT="${DT:-0.001}"
NX="${NX:-20}"
NY="${NY:-20}"
NZ="${NZ:-1}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"
MAX_FRAMES="${MAX_FRAMES:-60}"
THREADS="${THREADS:-1 2 3 4 5 6 7 8}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark_csv}"
CSV_FILE="${CSV_FILE:-$OUTPUT_DIR/numba_parallel_timings.csv}"

mkdir -p "$OUTPUT_DIR"
rm -f "$CSV_FILE"

for thread_count in $THREADS; do
    echo "running threads=$thread_count"
    NUMBA_NUM_THREADS="$thread_count" "$PYTHON_BIN" "$SCRIPT_DIR/nbodies_grid_numba.py" \
        "$DATASET" "$DT" "$NX" "$NY" "$NZ" \
        --warmup-frames "$WARMUP_FRAMES" \
        --max-frames "$MAX_FRAMES" \
        --benchmark-csv "$CSV_FILE"
done

echo "csv=$CSV_FILE"
