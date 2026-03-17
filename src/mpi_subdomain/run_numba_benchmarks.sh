#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_INTERPRETER="/home/santiago/OS202/Exam/.venv/bin/python"
MPI_RANKS="${MPI_RANKS:-2 4 6 8}"
THREADS="${THREADS:-4}"
MEASURED_FRAMES=60
CSV_FILE="$SCRIPT_DIR/benchmark_csv/mpi_subdomain_timings.csv"
DATASET_PATH="$SCRIPT_DIR/../data/galaxy_1000"

echo "MPI Subdomain Decomposition Benchmark Suite"
echo "============================================"
echo "Dataset: $DATASET_PATH"
echo "CSV Output: $CSV_FILE"
echo "Threads: $THREADS"
echo "MPI Ranks: $MPI_RANKS"
echo ""

mkdir -p "$(dirname "$CSV_FILE")"

for mpi_ranks in $MPI_RANKS; do
    export OMP_NUM_THREADS=$THREADS
    echo "running mpi_ranks=$mpi_ranks threads=$THREADS"
    
    mpiexec -n $mpi_ranks "$PYTHON_INTERPRETER" \
        "$SCRIPT_DIR/nbodies_grid_numba.py" \
        "$DATASET_PATH" \
        0.001 \
        20 20 1 \
        --warmup-frames 5 \
        --max-frames $((MEASURED_FRAMES + 5)) \
        --benchmark-csv "$CSV_FILE"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] mpi_ranks=$mpi_ranks completed"
    else
        echo "[ERROR] mpi_ranks=$mpi_ranks failed with exit code $?"
    fi
    echo ""
done

echo "Benchmark completed. Results saved to: $CSV_FILE"
