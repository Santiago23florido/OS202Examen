#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPIEXEC="${MPIEXEC:-mpiexec}"
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-/home/santiago/OS202/Exam/.venv/bin/python}"
MPI_RANKS="${MPI_RANKS:-1 2 3 4 5 6 7}"
WORKER_THREADS="${WORKER_THREADS:-2}"
MAX_CPU_BUDGET="${MAX_CPU_BUDGET:-16}"
ROOT_THREADS=1
MEASURED_FRAMES="${MEASURED_FRAMES:-60}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/benchmark_csv}"
DATASETS="${DATASETS:-${DATASET_PATH:-$SCRIPT_DIR/../data/galaxy_1000 $SCRIPT_DIR/../data/galaxy_10000}}"
DT="${DT:-0.001}"
NX="${NX:-20}"
NY="${NY:-20}"
NZ="${NZ:-1}"
GENERATE_PLOTS="${GENERATE_PLOTS:-1}"
PLOT_WORKER_THREADS="${PLOT_WORKER_THREADS:-2}"

echo "MPI Subdomain Decomposition Benchmark Suite"
echo "==========================================="
echo "Datasets: $DATASETS"
echo "MPI Ranks: $MPI_RANKS"
echo "Worker Threads: $WORKER_THREADS"
echo "CPU Budget: $MAX_CPU_BUDGET"
echo "Constraint: ${ROOT_THREADS} + (mpi_ranks - 1) * worker_threads <= $MAX_CPU_BUDGET"
echo ""

for dataset_path in $DATASETS; do
    dataset_label="$(basename "$dataset_path")"
    csv_file="${OUTPUT_DIR}/mpi_subdomain_timings_${dataset_label}.csv"
    plot_file="${OUTPUT_DIR}/mpi_subdomain_timings_${dataset_label}.png"

    mkdir -p "$(dirname "$csv_file")"
    rm -f "$csv_file"

    echo "MPI benchmark dataset=$dataset_path"
    echo "csv=$csv_file"

    for worker_threads in $WORKER_THREADS; do
        for mpi_ranks in $MPI_RANKS; do
            cpu_budget_used=$((ROOT_THREADS + (mpi_ranks - 1) * worker_threads))
            if [ "$cpu_budget_used" -gt "$MAX_CPU_BUDGET" ]; then
                echo "skipping dataset=$dataset_label mpi_ranks=$mpi_ranks worker_threads=$worker_threads cpu_budget=$cpu_budget_used"
                continue
            fi

            echo "running dataset=$dataset_label mpi_ranks=$mpi_ranks worker_threads=$worker_threads cpu_budget=$cpu_budget_used"

            BENCHMARK_WORKER_THREADS="$worker_threads" \
            NUMBA_NUM_THREADS="$worker_threads" \
            OMP_NUM_THREADS="$worker_threads" \
            MAX_CPU_BUDGET="$MAX_CPU_BUDGET" \
            "$MPIEXEC" --bind-to none -n "$mpi_ranks" "$PYTHON_INTERPRETER" \
                "$SCRIPT_DIR/nbodies_grid_numba.py" \
                "$dataset_path" \
                "$DT" \
                "$NX" "$NY" "$NZ" \
                --warmup-frames "$WARMUP_FRAMES" \
                --max-frames "$MEASURED_FRAMES" \
                --benchmark-csv "$csv_file"

            echo ""
        done
    done

    if [ "$GENERATE_PLOTS" = "1" ]; then
        "$PYTHON_INTERPRETER" "$SCRIPT_DIR/plot_benchmarks.py" \
            "$csv_file" \
            --output "$plot_file" \
            --worker-threads "$PLOT_WORKER_THREADS" \
            --mpi-min 1 \
            --mpi-max 7
    fi

    echo "completed dataset=$dataset_label"
    echo ""
done

echo "MPI benchmarks completed."
