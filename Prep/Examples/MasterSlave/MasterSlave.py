# Calcul de l'ensemble de Mandelbrot en Python avec strategie master-slave.
from math import log
from time import time

import matplotlib.cm
import numpy as np
from mpi4py import MPI
from PIL import Image

BATCH_SIZE = 10
WORK_TAG = 100
RESULT_META_TAG = 101
RESULT_DATA_TAG = 102
STOP_TAG = 103
ROOT = 0


class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def __contains__(self, c: complex) -> bool:
        sample = np.asarray([c], dtype=np.complex128)
        return self.count_iterations(sample)[0] == self.max_iterations

    def convergence(
        self,
        c: np.ndarray,
        smooth: bool = False,
        clamp: bool = True,
    ) -> np.ndarray:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray, smooth: bool = False) -> np.ndarray:
        iter_counts = np.full(c.shape, self.max_iterations, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c + 1.0) >= 0.25)
        z = np.zeros(c.shape, dtype=np.complex128)

        for it in range(self.max_iterations):
            if not np.any(mask):
                break

            z[mask] = z[mask] * z[mask] + c[mask]
            has_diverged = mask & (np.abs(z) > self.escape_radius)

            if np.any(has_diverged):
                iter_counts[has_diverged] = np.minimum(iter_counts[has_diverged], it)
                mask = mask & ~has_diverged

        if smooth:
            has_diverged = iter_counts < self.max_iterations
            iter_counts[has_diverged] += (
                1 - np.log(np.log(np.abs(z[has_diverged]))) / log(2)
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
) -> tuple[int, bool]:
    if next_row >= height:
        stop_payload = np.array([-1, 0], dtype=np.intc)
        comm.Send([stop_payload, MPI.INT], dest=worker, tag=STOP_TAG)
        return next_row, False

    row_count = min(batch_size, height - next_row)
    send_vec = np.array([next_row, row_count], dtype=np.intc)
    comm.Send([send_vec, MPI.INT], dest=worker, tag=WORK_TAG)
    return next_row + row_count, True


def master(
    comm,
    size: int,
    width: int,
    height: int,
    batch_size: int,
) -> np.ndarray:
    convergence = np.empty((width, height), dtype=np.double)
    next_row = 0
    active_workers = 0

    for worker in range(1, size):
        next_row, has_work = dispatch_work(comm, worker, next_row, height, batch_size)
        if has_work:
            active_workers += 1

    while active_workers > 0:
        status = MPI.Status()
        recv_vec = np.empty(2, dtype=np.intc)
        comm.Recv(
            [recv_vec, MPI.INT],
            source=MPI.ANY_SOURCE,
            tag=RESULT_META_TAG,
            status=status,
        )

        worker = status.source
        start_row = int(recv_vec[0])
        row_count = int(recv_vec[1])

        local_result = np.empty((width, row_count), dtype=np.double)
        comm.Recv([local_result, MPI.DOUBLE], source=worker, tag=RESULT_DATA_TAG)
        convergence[:, start_row : start_row + row_count] = local_result

        next_row, has_work = dispatch_work(comm, worker, next_row, height, batch_size)
        if not has_work:
            active_workers -= 1

    return convergence


def worker(
    comm,
    mandelbrot_set: MandelbrotSet,
    width: int,
    scale_x: float,
    scale_y: float,
) -> None:
    real_axis = -2.0 + scale_x * np.arange(width)

    while True:
        status = MPI.Status()
        recv_vec = np.empty(2, dtype=np.intc)
        comm.Recv([recv_vec, MPI.INT], source=ROOT, tag=MPI.ANY_TAG, status=status)

        if status.tag == STOP_TAG:
            break

        start_row = int(recv_vec[0])
        row_count = int(recv_vec[1])
        local_result = compute_rows(
            mandelbrot_set,
            real_axis,
            scale_y,
            start_row,
            row_count,
        )
        result_meta = np.array([start_row, row_count], dtype=np.intc)
        comm.Send([result_meta, MPI.INT], dest=ROOT, tag=RESULT_META_TAG)
        comm.Send([local_result, MPI.DOUBLE], dest=ROOT, tag=RESULT_DATA_TAG)


def render_image(convergence: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T) * 255))


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=2.0)
    width, height = 1024, 1024
    scale_x = 3.0 / width
    scale_y = 2.25 / height

    if size == 1:
        real_axis = -2.0 + scale_x * np.arange(width)
        start_time = time()
        convergence = compute_rows(mandelbrot_set, real_axis, scale_y, 0, height)
        end_time = time()
        print(f"Temps du calcul de l'ensemble de Mandelbrot : {end_time - start_time}")

        image_start = time()
        image = render_image(convergence)
        image_end = time()
        print(f"Temps de constitution de l'image : {image_end - image_start}")
        image.show()
        return

    if rank == ROOT:
        start_time = time()
        convergence = master(comm, size, width, height, BATCH_SIZE)
        end_time = time()
        print(f"Temps du calcul de l'ensemble de Mandelbrot : {end_time - start_time}")

        image_start = time()
        image = render_image(convergence)
        image_end = time()
        print(f"Temps de constitution de l'image : {image_end - image_start}")
        image.show()
    else:
        worker(comm, mandelbrot_set, width, scale_x, scale_y)


if __name__ == "__main__":
    main()
