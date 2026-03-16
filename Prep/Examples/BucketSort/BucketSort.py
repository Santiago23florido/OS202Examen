from time import time

import numpy as np
from mpi4py import MPI

ROOT = 0
VECTOR_SIZE = 1_000_000
RANDOM_SEED = 12345


def static_row_distribution(total_rows: int, nbp: int, rank: int) -> int:
    """Nombre d'elements attribues a ce rank."""
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


def static_dist(comm, size: int, rank: int, total_rows: int) -> np.ndarray:
    counts, displs = build_counts_displs(total_rows, size)
    recvcount = int(counts[rank])
    recvbuf = np.empty(recvcount, dtype=np.float64)

    if rank == ROOT:
        rng = np.random.default_rng(seed=RANDOM_SEED)
        global_vector = rng.uniform(0.0, 1.0, size=total_rows).astype(np.float64)
    else:
        global_vector = None

    comm.Scatterv(
        [global_vector, counts, displs, MPI.DOUBLE],
        [recvbuf, MPI.DOUBLE],
        root=ROOT,
    )
    return recvbuf


def local_regular_sample(local_vec: np.ndarray, size: int) -> np.ndarray:
    if local_vec.size == 0:
        return np.zeros(size, dtype=np.float64)

    sample_idx = np.linspace(0, local_vec.size - 1, size, dtype=int)
    return local_vec[sample_idx]


def box_distribution(comm, size: int, rank: int, sendbuf: np.ndarray) -> np.ndarray:
    recvcount = sendbuf.size

    if rank == ROOT:
        recvbuf = np.empty(size * recvcount, dtype=np.float64)
    else:
        recvbuf = None

    comm.Gather([sendbuf, MPI.DOUBLE], [recvbuf, recvcount, MPI.DOUBLE], root=ROOT)

    if rank == ROOT:
        recvbuf.sort()
        splitter_idx = np.linspace(0, recvbuf.size - 1, size + 1, dtype=int)[1:-1]
        splitters = recvbuf[splitter_idx].astype(np.float64, copy=False)
    else:
        splitters = np.empty(size - 1, dtype=np.float64)

    comm.Bcast([splitters, MPI.DOUBLE], root=ROOT)
    return splitters


def box_organize(
    comm,
    size: int,
    rank: int,
    boxes: np.ndarray,
    local_vec: np.ndarray,
) -> np.ndarray:
    del rank

    cuts = np.searchsorted(local_vec, boxes, side="right")
    starts = np.concatenate(([0], cuts)).astype(np.intc)
    ends = np.concatenate((cuts, [local_vec.size])).astype(np.intc)
    sendcounts = (ends - starts).astype(np.intc)
    sdispls = np.zeros(size, dtype=np.intc)
    sdispls[1:] = np.cumsum(sendcounts[:-1])

    recvcounts = np.empty(size, dtype=np.intc)
    comm.Alltoall([sendcounts, MPI.INT], [recvcounts, MPI.INT])

    rdispls = np.zeros(size, dtype=np.intc)
    rdispls[1:] = np.cumsum(recvcounts[:-1])
    recvbuf = np.empty(int(np.sum(recvcounts)), dtype=np.float64)

    comm.Alltoallv(
        [local_vec, sendcounts, sdispls, MPI.DOUBLE],
        [recvbuf, recvcounts, rdispls, MPI.DOUBLE],
    )

    recvbuf.sort()
    return recvbuf


def gather_sorted_vector(comm, size: int, rank: int, local_vec: np.ndarray) -> np.ndarray | None:
    local_count = int(local_vec.size)
    gathered_counts = comm.gather(local_count, root=ROOT)

    if rank == ROOT:
        recvcounts = np.array(gathered_counts, dtype=np.intc)
        displs = np.zeros(size, dtype=np.intc)
        displs[1:] = np.cumsum(recvcounts[:-1])
        global_vec = np.empty(int(np.sum(recvcounts)), dtype=np.float64)
    else:
        recvcounts = None
        displs = None
        global_vec = None

    comm.Gatherv(
        [local_vec, MPI.DOUBLE],
        [global_vec, recvcounts, displs, MPI.DOUBLE],
        root=ROOT,
    )
    return global_vec


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time()
    local_vec = static_dist(comm, size, rank, VECTOR_SIZE)
    local_vec.sort()

    local_sample = local_regular_sample(local_vec, size)
    boxes = box_distribution(comm, size, rank, local_sample)
    local_vec = box_organize(comm, size, rank, boxes, local_vec)
    global_sorted = gather_sorted_vector(comm, size, rank, local_vec)
    end_time = time()

    if rank == ROOT:
        is_sorted = bool(np.all(global_sorted[:-1] <= global_sorted[1:]))
        print(f"Temps du bucket sort parallele : {end_time - start_time}")
        print(f"Taille finale : {global_sorted.size}")
        print(f"Tri global correct : {is_sorted}")


if __name__ == "__main__":
    main()
