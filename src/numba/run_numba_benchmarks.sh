#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/../../.venv/bin/python}"
DATASETS="${DATASETS:-${DATASET:-$SCRIPT_DIR/../data/galaxy_1000 $SCRIPT_DIR/../data/galaxy_10000}}"
DT="${DT:-0.001}"
NX="${NX:-20}"
NY="${NY:-20}"
NZ="${NZ:-1}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"
MAX_FRAMES="${MAX_FRAMES:-60}"
THREADS="${THREADS:-1 2 3 4 5 6 7 8}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark_csv}"
GENERATE_PLOTS="${GENERATE_PLOTS:-1}"

mkdir -p "$OUTPUT_DIR"

for dataset_path in $DATASETS; do
    dataset_label="$(basename "$dataset_path")"
    csv_file="${OUTPUT_DIR}/numba_parallel_timings_${dataset_label}.csv"
    plot_file="${OUTPUT_DIR}/numba_parallel_timings_${dataset_label}.png"

    rm -f "$csv_file"

    echo "Numba benchmark dataset=$dataset_path"
    echo "csv=$csv_file"

    for thread_count in $THREADS; do
        echo "running dataset=$dataset_label threads=$thread_count"
        NUMBA_NUM_THREADS="$thread_count" "$PYTHON_BIN" "$SCRIPT_DIR/nbodies_grid_numba.py" \
            "$dataset_path" "$DT" "$NX" "$NY" "$NZ" \
            --warmup-frames "$WARMUP_FRAMES" \
            --max-frames "$MAX_FRAMES" \
            --benchmark-csv "$csv_file"
    done

    if [ "$GENERATE_PLOTS" = "1" ]; then
        "$PYTHON_BIN" "$SCRIPT_DIR/plot_benchmarks.py" "$csv_file" --output "$plot_file"
    fi

    echo "completed dataset=$dataset_label"
    echo ""
done

echo "Numba benchmarks completed. Outputs saved to: $OUTPUT_DIR"
