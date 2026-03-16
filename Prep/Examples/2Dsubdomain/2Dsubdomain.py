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


def static_distribution(total_size: int, parts: int, coord: int) -> int:
    base = total_size // parts
    rem = total_size % parts
    return base + 1 if coord < rem else base


def build_counts_displs(total_size: int, parts: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array(
        [static_distribution(total_size, parts, coord) for coord in range(parts)],
        dtype=np.intc,
    )
    displs = np.zeros(parts, dtype=np.intc)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


def choose_process_grid(size: int, shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = shape
    target_ratio = rows / cols
    candidates = []

    for proc_rows in range(1, size + 1):
        if size % proc_rows != 0:
            continue

        proc_cols = size // proc_rows
        if proc_rows > rows or proc_cols > cols:
            continue

        ratio = proc_rows / proc_cols
        score = abs(target_ratio - ratio)
        candidates.append((score, -min(rows // proc_rows, cols // proc_cols), proc_rows, proc_cols))

    if not candidates:
        raise ValueError("No se puede construir una grilla 2D de procesos valida para este tamano.")

    _, _, proc_rows, proc_cols = min(candidates)
    return proc_rows, proc_cols


class Grille:
    def __init__(self, dim, offset_row=0, offset_col=0, ghostcells=0, init_pattern=None):
        self.core_dim = dim
        self.offset_row = offset_row
        self.offset_col = offset_col
        self.ghostcells = ghostcells
        self.dimensions = (dim[0] + 2 * ghostcells, dim[1] + 2 * ghostcells)
        self.cells = np.zeros(self.dimensions, dtype=np.uint8)

        if init_pattern is not None:
            for row, col in init_pattern:
                local_row = row - offset_row + ghostcells
                local_col = col - offset_col + ghostcells
                in_rows = ghostcells <= local_row < ghostcells + dim[0]
                in_cols = ghostcells <= local_col < ghostcells + dim[1]
                if in_rows and in_cols:
                    self.cells[local_row, local_col] = 1
        else:
            rng = np.random.default_rng(seed=12345 + 1000 * offset_row + offset_col)
            if ghostcells == 0:
                self.cells = rng.integers(0, 2, size=self.dimensions, dtype=np.uint8)
            else:
                self.cells[ghostcells:-ghostcells, ghostcells:-ghostcells] = rng.integers(
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
        next_cells = np.zeros(self.dimensions, dtype=np.uint8)
        diff_cells = []
        c = np.ones((3, 3), dtype=np.int8)
        c[1, 1] = 0
        voisins = convolve2d(self.cells, c, mode="same", boundary="fill")

        if self.ghostcells == 0:
            core = self.cells
            voisins_core = voisins
            Grille.h(voisins_core)
            temp = core + voisins_core
            next_cells = np.clip(temp, 0, 1).astype(np.uint8)
        else:
            core = self.cells[self.ghostcells:-self.ghostcells, self.ghostcells:-self.ghostcells]
            voisins_core = voisins[self.ghostcells:-self.ghostcells, self.ghostcells:-self.ghostcells]
            Grille.h(voisins_core)
            temp = core + voisins_core
            next_cells[self.ghostcells:-self.ghostcells, self.ghostcells:-self.ghostcells] = np.clip(
                temp,
                0,
                1,
            ).astype(np.uint8)

        self.cells = next_cells
        return diff_cells

    def core_view(self) -> np.ndarray:
        if self.ghostcells == 0:
            return self.cells
        return self.cells[self.ghostcells:-self.ghostcells, self.ghostcells:-self.ghostcells]


def build_cart_comm(comm, shape: tuple[int, int]):
    dims = choose_process_grid(comm.Get_size(), shape)
    return comm.Create_cart(dims=dims, periods=[True, True], reorder=False)


def grid_distribution(cart_comm, shape: tuple[int, int], init_pattern=None):
    dims = cart_comm.Get_topo()[0]
    rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(rank)

    row_counts, row_displs = build_counts_displs(shape[0], dims[0])
    col_counts, col_displs = build_counts_displs(shape[1], dims[1])

    local_rows = int(row_counts[coords[0]])
    local_cols = int(col_counts[coords[1]])
    offset_row = int(row_displs[coords[0]])
    offset_col = int(col_displs[coords[1]])

    grille = Grille(
        (local_rows, local_cols),
        offset_row=offset_row,
        offset_col=offset_col,
        ghostcells=GHOSTS,
        init_pattern=init_pattern,
    )
    return grille, row_displs, col_displs


def get_neighbors(cart_comm):
    dims = cart_comm.Get_topo()[0]
    rank = cart_comm.Get_rank()
    row, col = cart_comm.Get_coords(rank)

    up = cart_comm.Get_cart_rank(((row - 1) % dims[0], col))
    down = cart_comm.Get_cart_rank(((row + 1) % dims[0], col))
    left = cart_comm.Get_cart_rank((row, (col - 1) % dims[1]))
    right = cart_comm.Get_cart_rank((row, (col + 1) % dims[1]))

    up_left = cart_comm.Get_cart_rank(((row - 1) % dims[0], (col - 1) % dims[1]))
    up_right = cart_comm.Get_cart_rank(((row - 1) % dims[0], (col + 1) % dims[1]))
    down_left = cart_comm.Get_cart_rank(((row + 1) % dims[0], (col - 1) % dims[1]))
    down_right = cart_comm.Get_cart_rank(((row + 1) % dims[0], (col + 1) % dims[1]))

    return {
        "up": up,
        "down": down,
        "left": left,
        "right": right,
        "up_left": up_left,
        "up_right": up_right,
        "down_left": down_left,
        "down_right": down_right,
    }


def exchange_halos(cart_comm, grille: Grille) -> None:
    if cart_comm.Get_size() == 1:
        grille.cells[0, 1:-1] = grille.cells[-2, 1:-1]
        grille.cells[-1, 1:-1] = grille.cells[1, 1:-1]
        grille.cells[1:-1, 0] = grille.cells[1:-1, -2]
        grille.cells[1:-1, -1] = grille.cells[1:-1, 1]
        grille.cells[0, 0] = grille.cells[-2, -2]
        grille.cells[0, -1] = grille.cells[-2, 1]
        grille.cells[-1, 0] = grille.cells[1, -2]
        grille.cells[-1, -1] = grille.cells[1, 1]
        return

    neighbors = get_neighbors(cart_comm)

    recv_bottom = np.empty(grille.core_dim[1], dtype=np.uint8)
    recv_top = np.empty(grille.core_dim[1], dtype=np.uint8)
    recv_right = np.empty(grille.core_dim[0], dtype=np.uint8)
    recv_left = np.empty(grille.core_dim[0], dtype=np.uint8)

    cart_comm.Sendrecv(
        sendbuf=[np.ascontiguousarray(grille.cells[GHOSTS, GHOSTS:-GHOSTS]), MPI.UNSIGNED_CHAR],
        dest=neighbors["up"],
        sendtag=10,
        recvbuf=[recv_bottom, MPI.UNSIGNED_CHAR],
        source=neighbors["down"],
        recvtag=10,
    )
    cart_comm.Sendrecv(
        sendbuf=[np.ascontiguousarray(grille.cells[-GHOSTS - 1, GHOSTS:-GHOSTS]), MPI.UNSIGNED_CHAR],
        dest=neighbors["down"],
        sendtag=11,
        recvbuf=[recv_top, MPI.UNSIGNED_CHAR],
        source=neighbors["up"],
        recvtag=11,
    )
    cart_comm.Sendrecv(
        sendbuf=[np.ascontiguousarray(grille.cells[GHOSTS:-GHOSTS, GHOSTS]), MPI.UNSIGNED_CHAR],
        dest=neighbors["left"],
        sendtag=12,
        recvbuf=[recv_right, MPI.UNSIGNED_CHAR],
        source=neighbors["right"],
        recvtag=12,
    )
    cart_comm.Sendrecv(
        sendbuf=[np.ascontiguousarray(grille.cells[GHOSTS:-GHOSTS, -GHOSTS - 1]), MPI.UNSIGNED_CHAR],
        dest=neighbors["right"],
        sendtag=13,
        recvbuf=[recv_left, MPI.UNSIGNED_CHAR],
        source=neighbors["left"],
        recvtag=13,
    )

    grille.cells[-1, GHOSTS:-GHOSTS] = recv_bottom
    grille.cells[0, GHOSTS:-GHOSTS] = recv_top
    grille.cells[GHOSTS:-GHOSTS, -1] = recv_right
    grille.cells[GHOSTS:-GHOSTS, 0] = recv_left

    send_top_left = np.array([grille.cells[GHOSTS, GHOSTS]], dtype=np.uint8)
    send_top_right = np.array([grille.cells[GHOSTS, -GHOSTS - 1]], dtype=np.uint8)
    send_bottom_left = np.array([grille.cells[-GHOSTS - 1, GHOSTS]], dtype=np.uint8)
    send_bottom_right = np.array([grille.cells[-GHOSTS - 1, -GHOSTS - 1]], dtype=np.uint8)

    recv_bottom_right = np.empty(1, dtype=np.uint8)
    recv_bottom_left = np.empty(1, dtype=np.uint8)
    recv_top_right = np.empty(1, dtype=np.uint8)
    recv_top_left = np.empty(1, dtype=np.uint8)

    cart_comm.Sendrecv(
        sendbuf=[send_top_left, MPI.UNSIGNED_CHAR],
        dest=neighbors["up_left"],
        sendtag=20,
        recvbuf=[recv_bottom_right, MPI.UNSIGNED_CHAR],
        source=neighbors["down_right"],
        recvtag=20,
    )
    cart_comm.Sendrecv(
        sendbuf=[send_top_right, MPI.UNSIGNED_CHAR],
        dest=neighbors["up_right"],
        sendtag=21,
        recvbuf=[recv_bottom_left, MPI.UNSIGNED_CHAR],
        source=neighbors["down_left"],
        recvtag=21,
    )
    cart_comm.Sendrecv(
        sendbuf=[send_bottom_left, MPI.UNSIGNED_CHAR],
        dest=neighbors["down_left"],
        sendtag=22,
        recvbuf=[recv_top_right, MPI.UNSIGNED_CHAR],
        source=neighbors["up_right"],
        recvtag=22,
    )
    cart_comm.Sendrecv(
        sendbuf=[send_bottom_right, MPI.UNSIGNED_CHAR],
        dest=neighbors["down_right"],
        sendtag=23,
        recvbuf=[recv_top_left, MPI.UNSIGNED_CHAR],
        source=neighbors["up_left"],
        recvtag=23,
    )

    grille.cells[-1, -1] = recv_bottom_right[0]
    grille.cells[-1, 0] = recv_bottom_left[0]
    grille.cells[0, -1] = recv_top_right[0]
    grille.cells[0, 0] = recv_top_left[0]


def gather_global_grid(cart_comm, grille: Grille, shape: tuple[int, int], row_displs, col_displs):
    rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(rank)
    payload = (coords, grille.core_view().copy())
    gathered = cart_comm.gather(payload, root=ROOT)

    if rank != ROOT:
        return None

    global_grid = np.zeros(shape, dtype=np.uint8)
    for block_coords, block in gathered:
        start_row = int(row_displs[block_coords[0]])
        start_col = int(col_displs[block_coords[1]])
        block_rows, block_cols = block.shape
        global_grid[start_row:start_row + block_rows, start_col:start_col + block_cols] = block

    return global_grid


def simulate_2d_subdomain(comm, shape: tuple[int, int], init_pattern, steps: int) -> np.ndarray | None:
    cart_comm = build_cart_comm(comm, shape)
    grille, row_displs, col_displs = grid_distribution(cart_comm, shape, init_pattern)

    for _ in range(steps):
        exchange_halos(cart_comm, grille)
        grille.compute_next_iteration()

    return gather_global_grid(cart_comm, grille, shape, row_displs, col_displs)


def simulate_serial(shape: tuple[int, int], init_pattern, steps: int) -> np.ndarray:
    grille = Grille(shape, 0, 0, 0, init_pattern)

    for _ in range(steps):
        c = np.ones((3, 3), dtype=np.int8)
        c[1, 1] = 0
        voisins = convolve2d(grille.cells, c, mode="same", boundary="wrap")
        Grille.h(voisins)
        temp = grille.cells + voisins
        grille.cells = np.clip(temp, 0, 1).astype(np.uint8)

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

    pattern_name, steps = parse_args()

    try:
        shape, init_pattern = PATTERNS[pattern_name]
    except KeyError as exc:
        if rank == ROOT:
            print(f"Pattern inconnu: {pattern_name}")
            print(f"Patterns disponibles: {sorted(PATTERNS.keys())}")
        raise SystemExit(1) from exc

    start_time = time()
    final_grid = simulate_2d_subdomain(comm, shape, init_pattern, steps)
    end_time = time()

    if rank == ROOT:
        proc_grid = choose_process_grid(comm.Get_size(), shape)
        alive_cells = int(np.sum(final_grid))
        print(f"Pattern initial choisi : {pattern_name}")
        print(f"Dimensions : {shape}")
        print(f"Grille de processus 2D : {proc_grid}")
        print(f"Nombre d'iterations : {steps}")
        print(f"Temps total 2D subdomain : {end_time - start_time}")
        print(f"Cellules vivantes finales : {alive_cells}")


if __name__ == "__main__":
    main()
