# Calcul de l'ensemble de Mandelbrot en Python avec strategie master-slave.
import argparse
import csv
from dataclasses import dataclass
from math import log
from pathlib import Path
import subprocess
from time import perf_counter

import matplotlib.cm
import numpy as np
from mpi4py import MPI
from PIL import Image

try:
    import numba
except ModuleNotFoundError:
    numba = None

BATCH_SIZE = 10
WORK_TAG = 100
RESULT_META_TAG = 101
RESULT_DATA_TAG = 102
STOP_TAG = 103
ROOT = 0
LOG_2 = log(2.0)
TIMINGS_CSV_PATH = Path(__file__).with_name("master_slave_numba_timings.csv")
HARDWARE_FIELDS = [
    "Architecture",
    "CPU op-mode(s)",
    "Address sizes",
    "Byte Order",
    "CPU(s)",
    "On-line CPU(s) list",
    "Vendor ID",
    "Model name",
    "CPU family",
    "Model",
    "Thread(s) per core",
    "Core(s) per socket",
    "Socket(s)",
    "Stepping",
    "BogoMIPS",
    "Virtualization",
    "Hypervisor vendor",
    "Virtualization type",
    "L1d cache",
    "L1i cache",
    "L2 cache",
    "L3 cache",
    "NUMA node(s)",
    "NUMA node0 CPU(s)",
]


@dataclass
class RankTiming:
    rank: int
    role: str
    compute_time: float = 0.0
    communication_time: float = 0.0

    def to_csv_row(self) -> dict[str, float | int | str]:
        return {
            "rank": self.rank,
            "role": self.role,
            "compute_time_seconds": self.compute_time,
            "communication_time_seconds": self.communication_time,
            "total_measured_time_seconds": self.compute_time + self.communication_time,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MPI master-slave de Mandelbrot.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=TIMINGS_CSV_PATH,
        help="Ruta del CSV de salida con los tiempos medidos.",
    )
    parser.add_argument(
        "--no-show-image",
        action="store_true",
        help="No abrir la imagen al final; util para benchmarks por lotes.",
    )
    parser.add_argument(
        "--no-print-hardware-info",
        action="store_true",
        help="No imprimir la tabla de hardware al arrancar.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Numero de columnas de la imagen de Mandelbrot.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Numero de filas de la imagen de Mandelbrot.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Numero maximo de iteraciones por punto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Numero de filas enviadas en cada lote MPI.",
    )
    return parser.parse_args()


def _parse_lscpu_output(raw_output: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in raw_output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def print_hardware_info() -> None:
    try:
        completed = subprocess.run(
            ["lscpu"],
            check=True,
            capture_output=True,
            text=True,
        )
        lscpu_data = _parse_lscpu_output(completed.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"No se pudo obtener la informacion de hardware con lscpu: {exc}")
        return

    champ_width = max(len("Champ"), max(len(field) for field in HARDWARE_FIELDS))
    print("Champ".ljust(champ_width), "Valeur")
    for field in HARDWARE_FIELDS:
        print(field.ljust(champ_width), lscpu_data.get(field, "N/A"))


def _count_iterations_python(
    c_arr: np.ndarray,
    max_iterations: int,
    escape_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    iter_counts = np.full(c_arr.shape, float(max_iterations), dtype=np.double)
    escape_values = np.full(c_arr.shape, escape_radius, dtype=np.double)

    for i in range(c_arr.shape[0]):
        for j in range(c_arr.shape[1]):
            c0 = c_arr[i, j]
            z = 0.0j
            abs_z = 0.0
            for it in range(max_iterations):
                z = z * z + c0
                abs_z = abs(z)
                if abs_z > escape_radius:
                    iter_counts[i, j] = float(it)
                    escape_values[i, j] = abs_z
                    break

    return iter_counts, escape_values


if numba is not None:

    @numba.njit(parallel=True)
    def _count_iterations_numba(
        c_arr: np.ndarray,
        max_iterations: int,
        escape_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        iter_counts = np.full(c_arr.shape, float(max_iterations), dtype=np.float64)
        escape_values = np.full(c_arr.shape, escape_radius, dtype=np.float64)

        for i in numba.prange(c_arr.shape[0]):
            for j in range(c_arr.shape[1]):
                c0 = c_arr[i, j]
                z = 0.0j
                abs_z = 0.0

                for it in range(max_iterations):
                    z = z * z + c0
                    abs_z = np.abs(z)
                    if abs_z > escape_radius:
                        iter_counts[i, j] = float(it)
                        escape_values[i, j] = abs_z
                        break

        return iter_counts, escape_values

else:

    def _count_iterations_numba(
        c_arr: np.ndarray,
        max_iterations: int,
        escape_radius: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _count_iterations_python(c_arr, max_iterations, escape_radius)


class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def __contains__(self, c: complex) -> bool:
        sample = np.asarray([[c]], dtype=np.complex128)
        return self.count_iterations(sample)[0, 0] == self.max_iterations

    def convergence(
        self,
        c: np.ndarray,
        smooth: bool = False,
        clamp: bool = True,
    ) -> np.ndarray:
        value = self.count_iterations(c, smooth=smooth) / self.max_iterations
        return np.clip(value, 0.0, 1.0) if clamp else value

    def count_iterations(self, c: np.ndarray, smooth: bool = False) -> np.ndarray:
        c_arr = np.asarray(c, dtype=np.complex128)
        if c_arr.ndim == 0:
            c_arr = c_arr.reshape(1, 1)
        elif c_arr.ndim == 1:
            c_arr = c_arr.reshape(c_arr.shape[0], 1)

        iter_counts, escape_values = _count_iterations_numba(
            c_arr,
            self.max_iterations,
            self.escape_radius,
        )

        if smooth:
            has_diverged = iter_counts < self.max_iterations
            if np.any(has_diverged):
                safe_escape_values = np.maximum(
                    escape_values[has_diverged],
                    self.escape_radius + 1e-12,
                )
                iter_counts[has_diverged] += (
                    1.0 - np.log(np.log(safe_escape_values)) / LOG_2
                )

        return iter_counts


def compute_rows(
    mandelbrot_set: MandelbrotSet,
    real_axis: np.ndarray,
    scale_y: float,
    start_row: int,
    row_count: int,
) -> np.ndarray:
    imag_axis = -1.125 + scale_y * np.arange(start_row, start_row + row_count)
    complex_grid = real_axis[:, np.newaxis] + 1j * imag_axis[np.newaxis, :]
    return mandelbrot_set.convergence(complex_grid, smooth=True)


def dispatch_work(
    comm,
    worker: int,
    next_row: int,
    height: int,
    batch_size: int,
) -> tuple[int, bool, float]:
    if next_row >= height:
        stop_payload = np.array([-1, 0], dtype=np.intc)
        # Tiempo de comunicacion: medimos solo el envio MPI del mensaje de parada.
        comm_start = perf_counter()
        comm.Send([stop_payload, MPI.INT], dest=worker, tag=STOP_TAG)
        return next_row, False, perf_counter() - comm_start

    row_count = min(batch_size, height - next_row)
    send_vec = np.array([next_row, row_count], dtype=np.intc)
    # Tiempo de comunicacion: medimos solo el envio MPI del siguiente bloque de trabajo.
    comm_start = perf_counter()
    comm.Send([send_vec, MPI.INT], dest=worker, tag=WORK_TAG)
    return next_row + row_count, True, perf_counter() - comm_start


def master(
    comm,
    size: int,
    width: int,
    height: int,
    batch_size: int,
) -> tuple[np.ndarray, RankTiming]:
    convergence = np.empty((width, height), dtype=np.double)
    next_row = 0
    active_workers = 0
    timings = RankTiming(rank=comm.Get_rank(), role="master")

    for worker in range(1, size):
        next_row, has_work, comm_time = dispatch_work(
            comm,
            worker,
            next_row,
            height,
            batch_size,
        )
        timings.communication_time += comm_time
        if has_work:
            active_workers += 1

    while active_workers > 0:
        status = MPI.Status()
        recv_vec = np.empty(2, dtype=np.intc)
        # Tiempo de comunicacion: medimos solo la recepcion MPI de los metadatos del resultado.
        comm_start = perf_counter()
        comm.Recv(
            [recv_vec, MPI.INT],
            source=MPI.ANY_SOURCE,
            tag=RESULT_META_TAG,
            status=status,
        )
        timings.communication_time += perf_counter() - comm_start

        worker = status.source
        start_row = int(recv_vec[0])
        row_count = int(recv_vec[1])

        local_result = np.empty((width, row_count), dtype=np.double)
        # Tiempo de comunicacion: medimos solo la recepcion MPI de la matriz calculada.
        comm_start = perf_counter()
        comm.Recv([local_result, MPI.DOUBLE], source=worker, tag=RESULT_DATA_TAG)
        timings.communication_time += perf_counter() - comm_start
        convergence[:, start_row : start_row + row_count] = local_result

        next_row, has_work, comm_time = dispatch_work(
            comm,
            worker,
            next_row,
            height,
            batch_size,
        )
        timings.communication_time += comm_time
        if not has_work:
            active_workers -= 1

    return convergence, timings


def worker(
    comm,
    mandelbrot_set: MandelbrotSet,
    width: int,
    scale_x: float,
    scale_y: float,
) -> RankTiming:
    real_axis = -2.0 + scale_x * np.arange(width)
    timings = RankTiming(rank=comm.Get_rank(), role="worker")

    while True:
        status = MPI.Status()
        recv_vec = np.empty(2, dtype=np.intc)
        # Tiempo de comunicacion: medimos solo la recepcion MPI del siguiente mensaje del master.
        comm_start = perf_counter()
        comm.Recv([recv_vec, MPI.INT], source=ROOT, tag=MPI.ANY_TAG, status=status)
        timings.communication_time += perf_counter() - comm_start

        if status.tag == STOP_TAG:
            break

        start_row = int(recv_vec[0])
        row_count = int(recv_vec[1])
        # Tiempo de calculo: medimos solo el algoritmo de Mandelbrot sobre el lote recibido.
        compute_start = perf_counter()
        local_result = compute_rows(
            mandelbrot_set,
            real_axis,
            scale_y,
            start_row,
            row_count,
        )
        timings.compute_time += perf_counter() - compute_start
        result_meta = np.array([start_row, row_count], dtype=np.intc)
        # Tiempo de comunicacion: medimos solo el envio MPI de los metadatos del resultado.
        comm_start = perf_counter()
        comm.Send([result_meta, MPI.INT], dest=ROOT, tag=RESULT_META_TAG)
        timings.communication_time += perf_counter() - comm_start
        # Tiempo de comunicacion: medimos solo el envio MPI de la matriz calculada.
        comm_start = perf_counter()
        comm.Send([local_result, MPI.DOUBLE], dest=ROOT, tag=RESULT_DATA_TAG)
        timings.communication_time += perf_counter() - comm_start

    return timings


def render_image(convergence: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))


def write_timings_csv(comm, timing: RankTiming, output_path: Path = TIMINGS_CSV_PATH) -> None:
    rows = comm.gather(timing.to_csv_row(), root=ROOT)
    if comm.Get_rank() != ROOT:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["rank"]))
    summary_row = {
        "rank": "ALL",
        "role": "aggregate_sum",
        "compute_time_seconds": sum(float(row["compute_time_seconds"]) for row in rows),
        "communication_time_seconds": sum(float(row["communication_time_seconds"]) for row in rows),
        "total_measured_time_seconds": sum(float(row["total_measured_time_seconds"]) for row in rows),
    }

    with output_path.open("w", newline="", encoding="ascii") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "role",
                "compute_time_seconds",
                "communication_time_seconds",
                "total_measured_time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary_row)

    print(f"CSV de tiempos guardado en: {output_path}")
    print("Resumen tiempos:")
    print(f"  calculo total      : {summary_row['compute_time_seconds']:.6f} s")
    print(f"  comunicacion total : {summary_row['communication_time_seconds']:.6f} s")
    print(f"  medido total       : {summary_row['total_measured_time_seconds']:.6f} s")


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    width, height = args.width, args.height
    scale_x = 3.0 / width
    scale_y = 2.25 / height
    mandelbrot_set = MandelbrotSet(max_iterations=args.max_iterations, escape_radius=2.0)

    if rank == ROOT and numba is None:
        print("Numba no esta instalado; usando el fallback en Python.")

    if rank == ROOT and not args.no_print_hardware_info:
        print_hardware_info()

    if size == 1:
        real_axis = -2.0 + scale_x * np.arange(width)
        timings = RankTiming(rank=rank, role="sequential")
        # Tiempo de calculo: medimos solo el algoritmo secuencial de Mandelbrot.
        compute_start = perf_counter()
        convergence = compute_rows(mandelbrot_set, real_axis, scale_y, 0, height)
        timings.compute_time += perf_counter() - compute_start
        write_timings_csv(comm, timings, args.csv_path)
        image = render_image(convergence)
        if not args.no_show_image:
            image.show()
        return

    if rank == ROOT:
        convergence, timings = master(comm, size, width, height, args.batch_size)
        write_timings_csv(comm, timings, args.csv_path)
        image = render_image(convergence)
        if not args.no_show_image:
            image.show()
    else:
        timings = worker(comm, mandelbrot_set, width, scale_x, scale_y)
        write_timings_csv(comm, timings, args.csv_path)


if __name__ == "__main__":
    main()
