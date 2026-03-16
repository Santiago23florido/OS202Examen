#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_script="${script_dir}/MasterSlaveNumba.py"
plot_script="${script_dir}/plot_master_slave_numba_timings.py"
results_dir="${script_dir}/benchmark_csv"

mkdir -p "${results_dir}"

if command -v mpiexec >/dev/null 2>&1; then
  mpi_launcher="mpiexec"
elif command -v mpirun >/dev/null 2>&1; then
  mpi_launcher="mpirun"
else
  echo "No se encontro mpiexec ni mpirun en el sistema." >&2
  exit 1
fi

# Evita sobrecarga excesiva cuando se combinan multiples procesos MPI con Numba.
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"

process_counts=(2 4 6 8)
print_hardware_info="yes"

for process_count in "${process_counts[@]}"; do
  csv_path="${results_dir}/master_slave_numba_${process_count}p.csv"
  echo "Ejecutando benchmark con ${process_count} procesos..."

  extra_args=(
    "--csv-path" "${csv_path}"
    "--no-show-image"
  )

  if [[ "${print_hardware_info}" == "no" ]]; then
    extra_args+=("--no-print-hardware-info")
  fi

  "${mpi_launcher}" -n "${process_count}" python3 "${python_script}" "${extra_args[@]}" "$@"
  print_hardware_info="no"
done

python3 "${plot_script}" --input-dir "${results_dir}"
