# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.
from pathlib import Path
from time import time

import numpy as np
from PIL import Image
from mpi4py import MPI
from scipy import signal

ROOT = 0
GHOSTS = 1


def static_row_distribution(total_rows: int, nbp: int, rank: int) -> int:
    base = total_rows // nbp
    rem = total_rows % nbp
    return base + 1 if rank < rem else base


def build_counts_displs(total_rows: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array(
        [static_row_distribution(total_rows, size, rank) for rank in range(size)],
        dtype=np.intc,
    )
    displs = np.zeros(size, dtype=np.intc)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


class Grille:
    def __init__(self, dim, offset_row=0, ghostcells=0):
        self.core_dim = dim
        self.offset_row = offset_row
        self.ghostcells = ghostcells
        self.dimensions = (dim[0] + 2 * ghostcells, dim[1], 3)
        self.cells = np.zeros(self.dimensions, dtype=np.double)

    def compute_next_iteration(self):
        next_cells = np.zeros(self.dimensions, dtype=np.double)
        mask = np.array(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=np.double,
        ) / 16.0
        blur_image = np.zeros_like(self.cells, dtype=np.double)
        for i in range(3):
            blur_image[:, :, i] = signal.convolve2d(
                self.cells[:, :, i],
                mask,
                mode="same",
            )

        mask = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=np.double,
        )
        sharpen_image = np.zeros_like(self.cells, dtype=np.double)
        sharpen_image[:, :, :2] = blur_image[:, :, :2]
        sharpen_image[:, :, 2] = np.clip(
            signal.convolve2d(blur_image[:, :, 2], mask, mode="same"),
            0.0,
            1.0,
        )
        next_cells = sharpen_image

        if self.ghostcells == 0:
            self.cells = next_cells
        else:
            self.cells[self.ghostcells:-self.ghostcells, :, :] = next_cells[
                self.ghostcells:-self.ghostcells, :, :
            ]
        return self.cells

    def core_view(self) -> np.ndarray:
        if self.ghostcells == 0:
            return self.cells
        return self.cells[self.ghostcells:-self.ghostcells, :, :]


def load_image(image: str) -> np.ndarray:
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    img = img.convert("HSV")
    img = np.array(img, dtype=np.double)
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.0
    print(f"Nouvelle taille : {img.shape}")
    return img


def grid_distribution(size: int, rank: int, image: str, comm) -> tuple[Grille, int, int]:
    if rank == ROOT:
        full_image = load_image(image)
        total_rows, total_cols = full_image.shape[:2]
    else:
        full_image = None
        total_rows = 0
        total_cols = 0

    total_rows, total_cols = comm.bcast((total_rows, total_cols), root=ROOT)
    counts, displs = build_counts_displs(total_rows, size)
    local_rows = int(counts[rank])
    offset_row = int(displs[rank])
    grid = Grille((local_rows, total_cols), offset_row, GHOSTS)

    local_core = grid.core_view()
    if rank == ROOT:
        sendcounts = (counts * total_cols * 3).astype(np.intc)
        senddispls = (displs * total_cols * 3).astype(np.intc)
    else:
        sendcounts = None
        senddispls = None

    comm.Scatterv(
        [full_image, sendcounts, senddispls, MPI.DOUBLE],
        [local_core, MPI.DOUBLE],
        root=ROOT,
    )
    return grid, total_rows, total_cols


def exchange_ghost_rows(comm, rank: int, size: int, grille: Grille) -> None:
    grille.cells[0, :, :] = 0.0
    grille.cells[-1, :, :] = 0.0

    prev_rank = rank - 1 if rank > 0 else MPI.PROC_NULL
    next_rank = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    comm.Sendrecv(
        sendbuf=[grille.cells[GHOSTS, :, :], MPI.DOUBLE],
        dest=prev_rank,
        sendtag=10,
        recvbuf=[grille.cells[-GHOSTS, :, :], MPI.DOUBLE],
        source=next_rank,
        recvtag=10,
    )
    comm.Sendrecv(
        sendbuf=[grille.cells[-GHOSTS - 1, :, :], MPI.DOUBLE],
        dest=next_rank,
        sendtag=11,
        recvbuf=[grille.cells[0, :, :], MPI.DOUBLE],
        source=prev_rank,
        recvtag=11,
    )


def gather_grid_rows(
    comm,
    rank: int,
    size: int,
    grille: Grille,
    total_rows: int,
    total_cols: int,
) -> np.ndarray | None:
    counts, displs = build_counts_displs(total_rows, size)
    local_core = grille.core_view()

    if rank == ROOT:
        recvcounts = (counts * total_cols * 3).astype(np.intc)
        recvdispls = (displs * total_cols * 3).astype(np.intc)
        global_grid = np.empty((total_rows, total_cols, 3), dtype=np.double)
    else:
        recvcounts = None
        recvdispls = None
        global_grid = None

    comm.Gatherv(
        [local_core, MPI.DOUBLE],
        [global_grid, recvcounts, recvdispls, MPI.DOUBLE],
        root=ROOT,
    )
    return global_grid


def simulate_rows_subdomain(comm, rank: int, size: int, image: str) -> np.ndarray | None:
    grille, total_rows, total_cols = grid_distribution(size, rank, image, comm)
    exchange_ghost_rows(comm, rank, size, grille)
    grille.compute_next_iteration()
    return gather_grid_rows(comm, rank, size, grille, total_rows, total_cols)


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    image = "datas/paysage.jpg"
    output_path = Path("sorties/paysage_double.jpg")
    if rank == ROOT:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time()
    final_grid = simulate_rows_subdomain(comm, rank, size, image)
    end_time = time()

    if rank == ROOT:
        doubled_image = Image.fromarray(
            (255.0 * np.clip(final_grid, 0.0, 1.0)).astype(np.uint8),
            "HSV",
        ).convert("RGB")
        doubled_image.save(output_path)
        print(f"Temps total : {end_time - start_time}")
        print("Image sauvegardee")


if __name__ == "__main__":
    main()
