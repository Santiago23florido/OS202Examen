import sys
from time import time

import numpy as np
from mpi4py import MPI
from scipy.signal import convolve2d

ROOT = 0
GHOSTS = 1
DEFAULT_PATTERN = "glider"
DEFAULT_STEPS = 200

PATTERNS = {
    "blinker": ((5, 5), [(2, 1), (2, 2), (2, 3)]),
    "toad": ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
    "acorn": (
        (100, 100),
        [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)],
    ),
    "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
    "boat": ((5, 5), [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]),
    "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
    "glider_gun": (
        (400, 400),
        [
            (51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73),
            (53, 86), (53, 87), (54, 63), (54, 67), (54, 72), (54, 73), (54, 86),
            (54, 87), (55, 52), (55, 53), (55, 62), (55, 68), (55, 72), (55, 73),
            (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69), (56, 74),
            (56, 76), (57, 62), (57, 68), (57, 76), (58, 63), (58, 67), (59, 64),
            (59, 65),
        ],
    ),
    "space_ship": (
        (25, 25),
        [(11, 13), (11, 14), (12, 11), (12, 12), (12, 14), (12, 15), (13, 11),
         (13, 12), (13, 13), (13, 14), (14, 12), (14, 13)],
    ),
    "die_hard": ((100, 100), [(51, 57), (52, 51), (52, 52), (53, 52), (53, 56), (53, 57), (53, 58)]),
    "pulsar": (
        (17, 17),
        [
            (2, 4), (2, 5), (2, 6), (7, 4), (7, 5), (7, 6), (9, 4), (9, 5), (9, 6),
            (14, 4), (14, 5), (14, 6), (2, 10), (2, 11), (2, 12), (7, 10), (7, 11),
            (7, 12), (9, 10), (9, 11), (9, 12), (14, 10), (14, 11), (14, 12), (4, 2),
            (5, 2), (6, 2), (4, 7), (5, 7), (6, 7), (4, 9), (5, 9), (6, 9), (4, 14),
            (5, 14), (6, 14), (10, 2), (11, 2), (12, 2), (10, 7), (11, 7), (12, 7),
            (10, 9), (11, 9), (12, 9), (10, 14), (11, 14), (12, 14),
        ],
    ),
    "floraison": ((40, 40), [(19, 18), (19, 19), (19, 20), (20, 17), (20, 19), (20, 21), (21, 18), (21, 19), (21, 20)]),
    "block_switch_engine": (
        (400, 400),
        [(201, 202), (201, 203), (202, 202), (202, 203), (211, 203), (212, 204),
         (212, 202), (214, 204), (214, 201), (215, 201), (215, 202), (216, 201)],
    ),
    "u": (
        (200, 200),
        [(101, 101), (102, 102), (103, 102), (103, 101), (104, 103), (105, 103),
         (105, 102), (105, 101), (105, 105), (103, 105), (102, 105), (101, 105), (101, 104)],
    ),
    "flat": (
        (200, 400),
        [
            (80, 200), (81, 200), (82, 200), (83, 200), (84, 200), (85, 200), (86, 200),
            (87, 200), (89, 200), (90, 200), (91, 200), (92, 200), (93, 200), (97, 200),
            (98, 200), (99, 200), (106, 200), (107, 200), (108, 200), (109, 200), (110, 200),
            (111, 200), (112, 200), (114, 200), (115, 200), (116, 200), (117, 200), (118, 200),
        ],
    ),
}


def static_row_distribution(total_rows: int, nbp: int, rank: int) -> int:
    """Nombre de lignes attribuees a ce rank."""
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
    def __init__(self, dim, offset_row=0, ghostcells=0, init_pattern=None):
        self.core_dim = dim
        self.offset_row = offset_row
        self.ghostcells = ghostcells
        self.dimensions = (dim[0] + 2 * ghostcells, dim[1])

        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        if init_pattern is not None:
            for row, col in init_pattern:
                local_row = row - offset_row + ghostcells
                in_local_rows = ghostcells <= local_row < ghostcells + dim[0]
                if in_local_rows and 0 <= col < dim[1]:
                    self.cells[local_row, col] = 1
        else:
            rng = np.random.default_rng(seed=12345 + offset_row)
            if ghostcells == 0:
                self.cells = rng.integers(0, 2, size=self.dimensions, dtype=np.uint8)
            else:
                self.cells[ghostcells:-ghostcells, :] = rng.integers(
                    0,
                    2,
                    size=dim,
                    dtype=np.uint8,
                )

    @staticmethod
    def h(x):
        x[x <= 1] = -1
        x[x >= 4] = -1
        x[x == 2] = 0
        x[x == 3] = 1

    def compute_next_iteration(self):
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        diff_cells = []
        c = np.ones((3, 3))
        c[1, 1] = 0
        voisins = convolve2d(self.cells, c, mode="same", boundary="wrap")
        Grille.h(voisins)
        temp = self.cells + voisins
        next_cells = np.clip(temp, 0, 1)

        self.cells = next_cells
        return diff_cells

    def core_view(self) -> np.ndarray:
        if self.ghostcells == 0:
            return self.cells
        return self.cells[self.ghostcells:-self.ghostcells, :]


def grid_distribution(size: int, rank: int, shape: tuple[int, int], init_pattern=None) -> Grille:
    counts, displs = build_counts_displs(shape[0], size)
    local_rows = int(counts[rank])
    offset_row = int(displs[rank])
    return Grille((local_rows, shape[1]), offset_row, GHOSTS, init_pattern)


def exchange_ghost_rows(comm, rank: int, size: int, grille: Grille) -> None:
    if size == 1:
        grille.cells[0, :] = grille.cells[-2, :]
        grille.cells[-1, :] = grille.cells[1, :]
        return

    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    comm.Sendrecv(
        sendbuf=[grille.cells[GHOSTS, :], MPI.UNSIGNED_CHAR],
        dest=prev_rank,
        sendtag=10,
        recvbuf=[grille.cells[-GHOSTS, :], MPI.UNSIGNED_CHAR],
        source=next_rank,
        recvtag=10,
    )
    comm.Sendrecv(
        sendbuf=[grille.cells[-GHOSTS - 1, :], MPI.UNSIGNED_CHAR],
        dest=next_rank,
        sendtag=11,
        recvbuf=[grille.cells[0, :], MPI.UNSIGNED_CHAR],
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
    local_core = grille.core_view().astype(np.uint8, copy=False)

    if rank == ROOT:
        recvcounts = (counts * total_cols).astype(np.intc)
        recvdispls = (displs * total_cols).astype(np.intc)
        global_grid = np.empty((total_rows, total_cols), dtype=np.uint8)
    else:
        recvcounts = None
        recvdispls = None
        global_grid = None

    comm.Gatherv(
        [local_core, MPI.UNSIGNED_CHAR],
        [global_grid, recvcounts, recvdispls, MPI.UNSIGNED_CHAR],
        root=ROOT,
    )
    return global_grid


def simulate_rows_subdomain(
    comm,
    rank: int,
    size: int,
    shape: tuple[int, int],
    init_pattern,
    steps: int,
) -> np.ndarray | None:
    grille = grid_distribution(size, rank, shape, init_pattern)

    for _ in range(steps):
        exchange_ghost_rows(comm, rank, size, grille)
        grille.compute_next_iteration()
        grille.cells = grille.cells.astype(np.uint8, copy=False)

    return gather_grid_rows(comm, rank, size, grille, shape[0], shape[1])


def simulate_serial(shape: tuple[int, int], init_pattern, steps: int) -> np.ndarray:
    grille = Grille(shape, 0, 0, init_pattern)

    for _ in range(steps):
        grille.compute_next_iteration()
        grille.cells = grille.cells.astype(np.uint8, copy=False)

    return grille.cells.copy()


def parse_args() -> tuple[str, int]:
    pattern_name = DEFAULT_PATTERN
    steps = DEFAULT_STEPS

    if len(sys.argv) > 1:
        pattern_name = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])

    return pattern_name, steps


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    pattern_name, steps = parse_args()

    try:
        shape, init_pattern = PATTERNS[pattern_name]
    except KeyError as exc:
        if rank == ROOT:
            print(f"Pattern inconnu: {pattern_name}")
            print(f"Patterns disponibles: {sorted(PATTERNS.keys())}")
        raise SystemExit(1) from exc

    if size > shape[0]:
        if rank == ROOT:
            print("Le nombre de processus ne peut pas exceder le nombre de lignes.")
        raise SystemExit(1)

    start_time = time()
    final_grid = simulate_rows_subdomain(comm, rank, size, shape, init_pattern, steps)
    end_time = time()

    if rank == ROOT:
        alive_cells = int(np.sum(final_grid))
        print(f"Pattern initial choisi : {pattern_name}")
        print(f"Dimensions : {shape}")
        print(f"Nombre d'iterations : {steps}")
        print(f"Temps total rows subdomain : {end_time - start_time}")
        print(f"Cellules vivantes finales : {alive_cells}")


if __name__ == "__main__":
    main()
