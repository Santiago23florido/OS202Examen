import argparse
import csv
import os
import time
from itertools import product

import numpy as np
from mpi4py import MPI
from numba import get_num_threads, njit, prange, set_num_threads

import visualizer3d

G = 1.560339e-13
GHOST_LAYER = 2
ROOT_WORLD_RANK = 0
DEFAULT_MAX_CPU_BUDGET = 16
WORKER_BREAKDOWN_KEYS = (
    "worker_accel_pre_ms",
    "worker_position_update_ms",
    "worker_migration_ms",
    "worker_global_grid_ms",
    "worker_ghost_exchange_ms",
    "worker_local_grid_ms",
    "worker_accel_post_ms",
    "worker_velocity_update_ms",
)


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


def choose_process_grid(size: int, nb_cells_per_dim: tuple[int, int, int]) -> tuple[int, int, int]:
    """Encuentra una descomposición 3D sin sobrepasar las celdas disponibles."""
    best = None
    best_volume = None
    nx, ny, nz = nb_cells_per_dim
    for px in range(1, size + 1):
        if size % px != 0 or px > nx:
            continue
        rem = size // px
        for py in range(1, rem + 1):
            if rem % py != 0 or py > ny:
                continue
            pz = rem // py
            if pz > nz:
                continue
            volume = max(px, py, pz)
            if best is None or volume < best_volume:
                best = (px, py, pz)
                best_volume = volume
    if best is None:
        raise ValueError(
            f"No hay descomposición MPI válida para size={size} y grid={nb_cells_per_dim}"
        )
    return best


def load_dataset(filename):
    positions = []
    velocities = []
    masses = []
    max_mass = 0.0
    box = np.array(
        [[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]],
        dtype=np.float32,
    )
    with open(filename, "r") as handle:
        line = handle.readline()
        while line:
            data = line.split()
            mass = float(data[0])
            position = [float(data[1]), float(data[2]), float(data[3])]
            velocity = [float(data[4]), float(data[5]), float(data[6])]
            masses.append(mass)
            positions.append(position)
            velocities.append(velocity)
            max_mass = max(max_mass, mass)
            for axis in range(3):
                box[0][axis] = min(box[0][axis], position[axis] - 1.0e-6)
                box[1][axis] = max(box[1][axis], position[axis] + 1.0e-6)
            line = handle.readline()
    positions = np.array(positions, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    masses = np.array(masses, dtype=np.float32)
    colors = [generate_star_color(mass) for mass in masses]
    return positions, velocities, masses, box, max_mass, colors


def get_requested_worker_threads() -> int:
    value = os.environ.get("BENCHMARK_WORKER_THREADS") or os.environ.get("NUMBA_NUM_THREADS")
    if value is None:
        return max(1, get_num_threads())
    return max(1, int(value))


def get_effective_cpu_budget() -> int:
    cpu_count = os.cpu_count() or DEFAULT_MAX_CPU_BUDGET
    configured_budget = int(os.environ.get("MAX_CPU_BUDGET", str(DEFAULT_MAX_CPU_BUDGET)))
    return max(1, min(cpu_count, configured_budget))


def required_cpu_budget(mpi_ranks: int, worker_threads: int) -> int:
    if mpi_ranks <= 1:
        return 1
    return 1 + (mpi_ranks - 1) * worker_threads


def configure_rank_resources(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    worker_threads = get_requested_worker_threads()
    cpu_budget = get_effective_cpu_budget()
    required_budget = required_cpu_budget(size, worker_threads)
    if required_budget > cpu_budget:
        message = (
            f"Configuracion invalida: mpi_ranks={size}, worker_threads={worker_threads}, "
            f"cpu_budget_required={required_budget}, cpu_budget_available={cpu_budget}"
        )
        if rank == ROOT_WORLD_RANK:
            raise SystemExit(message)
        raise SystemExit(0)

    assigned_cpus = {0}
    if rank == ROOT_WORLD_RANK:
        assigned_cpus = {0}
        set_num_threads(1)
    else:
        start_cpu = 1 + (rank - 1) * worker_threads
        stop_cpu = start_cpu + worker_threads
        assigned_cpus = set(range(start_cpu, stop_cpu))
        set_num_threads(worker_threads)

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, assigned_cpus)
        assigned_cpus = set(os.sched_getaffinity(0))

    return {
        "worker_threads": worker_threads,
        "root_threads": 1,
        "cpu_budget_used": required_budget,
        "assigned_cpus": sorted(assigned_cpus),
    }


def write_benchmark_row(csv_path, row):
    if csv_path is None:
        return
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_metadata(filename, dt, ncells_per_dir, mpi_ranks, worker_threads, root_threads, cpu_budget_used):
    return {
        "threads": str(worker_threads),
        "worker_threads": str(worker_threads),
        "root_threads": str(root_threads),
        "worker_ranks": str(max(0, mpi_ranks - 1)),
        "mpi_ranks": str(mpi_ranks),
        "cpu_budget_used": str(cpu_budget_used),
        "filename": os.path.abspath(filename),
        "dt": str(dt),
        "nx": str(ncells_per_dir[0]),
        "ny": str(ncells_per_dir[1]),
        "nz": str(ncells_per_dir[2]),
    }


def zero_worker_breakdown():
    return {key: 0.0 for key in WORKER_BREAKDOWN_KEYS}


def summarize_worker_breakdown(
    avg_worker_compute_ms,
    avg_worker_send_positions_ms,
    avg_root_step_ms,
    avg_root_wait_result_ms,
    breakdown,
):
    avg_worker_force_ms = breakdown["worker_accel_pre_ms"] + breakdown["worker_accel_post_ms"]
    avg_worker_integration_ms = (
        breakdown["worker_position_update_ms"] + breakdown["worker_velocity_update_ms"]
    )
    avg_worker_sync_ms = (
        breakdown["worker_migration_ms"]
        + breakdown["worker_global_grid_ms"]
        + breakdown["worker_ghost_exchange_ms"]
    )
    avg_worker_step_accounted_ms = (
        avg_worker_force_ms
        + avg_worker_integration_ms
        + avg_worker_sync_ms
        + breakdown["worker_local_grid_ms"]
    )
    avg_worker_unaccounted_ms = max(0.0, avg_worker_compute_ms - avg_worker_step_accounted_ms)
    pct_worker_force_of_compute = 0.0
    pct_worker_sync_of_compute = 0.0
    pct_worker_send_of_step = 0.0
    pct_root_wait_of_step = 0.0
    if avg_worker_compute_ms > 0.0:
        pct_worker_force_of_compute = 100.0 * avg_worker_force_ms / avg_worker_compute_ms
        pct_worker_sync_of_compute = 100.0 * avg_worker_sync_ms / avg_worker_compute_ms
    if avg_root_step_ms > 0.0:
        pct_worker_send_of_step = 100.0 * avg_worker_send_positions_ms / avg_root_step_ms
        pct_root_wait_of_step = 100.0 * avg_root_wait_result_ms / avg_root_step_ms
    return {
        "avg_worker_accel_pre_ms": breakdown["worker_accel_pre_ms"],
        "avg_worker_position_update_ms": breakdown["worker_position_update_ms"],
        "avg_worker_migration_ms": breakdown["worker_migration_ms"],
        "avg_worker_global_grid_ms": breakdown["worker_global_grid_ms"],
        "avg_worker_ghost_exchange_ms": breakdown["worker_ghost_exchange_ms"],
        "avg_worker_local_grid_ms": breakdown["worker_local_grid_ms"],
        "avg_worker_accel_post_ms": breakdown["worker_accel_post_ms"],
        "avg_worker_velocity_update_ms": breakdown["worker_velocity_update_ms"],
        "avg_worker_force_ms": avg_worker_force_ms,
        "avg_worker_integration_ms": avg_worker_integration_ms,
        "avg_worker_sync_ms": avg_worker_sync_ms,
        "avg_worker_step_accounted_ms": avg_worker_step_accounted_ms,
        "avg_worker_unaccounted_ms": avg_worker_unaccounted_ms,
        "pct_worker_force_of_compute": pct_worker_force_of_compute,
        "pct_worker_sync_of_compute": pct_worker_sync_of_compute,
        "pct_worker_send_of_step": pct_worker_send_of_step,
        "pct_root_wait_of_step": pct_root_wait_of_step,
    }


def build_counts_and_displacements(counts):
    counts = np.array(counts, dtype=np.int32)
    displacements = np.zeros_like(counts)
    if counts.shape[0] > 1:
        displacements[1:] = np.cumsum(counts[:-1])
    return counts, displacements


def gatherv_1d_to_root(comm, local_array, mpi_dtype):
    rank = comm.Get_rank()
    counts = comm.gather(int(local_array.shape[0]), root=0)
    recv_array = None
    recv_spec = None
    if rank == 0:
        counts, displacements = build_counts_and_displacements(counts)
        recv_array = np.empty(int(np.sum(counts)), dtype=local_array.dtype)
        recv_spec = [recv_array, counts, displacements, mpi_dtype]
    comm.Gatherv([np.ascontiguousarray(local_array), mpi_dtype], recv_spec, root=0)
    return recv_array


def gatherv_vec3_to_root(comm, local_array, mpi_dtype):
    rank = comm.Get_rank()
    counts = comm.gather(int(local_array.shape[0]), root=0)
    recv_array = None
    recv_spec = None
    if rank == 0:
        counts, displacements = build_counts_and_displacements(counts)
        recv_counts = counts * 3
        recv_displacements = displacements * 3
        recv_array = np.empty(int(np.sum(recv_counts)), dtype=local_array.dtype)
        recv_spec = [recv_array, recv_counts, recv_displacements, mpi_dtype]
    comm.Gatherv([np.ascontiguousarray(local_array.reshape(-1)), mpi_dtype], recv_spec, root=0)
    if rank == 0:
        return recv_array.reshape((-1, 3))
    return None


def take_particle_payload(
    ids,
    positions,
    velocities,
    masses,
    previous_accelerations,
    indices,
):
    if len(indices) == 0:
        return empty_migration_payload()
    selected = np.array(indices, dtype=np.int64)
    return (
        ids[selected].copy(),
        positions[selected].copy(),
        velocities[selected].copy(),
        masses[selected].copy(),
        previous_accelerations[selected].copy(),
    )


def empty_migration_payload():
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )


def empty_ghost_payload():
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )


def _concat_or_empty(chunks, dtype, ndim=1):
    if not chunks:
        if ndim == 1:
            return np.empty(0, dtype=dtype)
        return np.empty((0, ndim), dtype=dtype)
    if ndim == 1:
        return np.concatenate(chunks).astype(dtype, copy=False)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)


def merge_migration_payloads(payloads):
    ids_chunks = []
    pos_chunks = []
    vel_chunks = []
    mass_chunks = []
    accel_chunks = []
    for ids, positions, velocities, masses, previous_accelerations in payloads:
        if ids.size == 0:
            continue
        ids_chunks.append(ids)
        pos_chunks.append(positions)
        vel_chunks.append(velocities)
        mass_chunks.append(masses)
        accel_chunks.append(previous_accelerations)
    if not ids_chunks:
        return empty_migration_payload()
    ids = _concat_or_empty(ids_chunks, np.int64)
    positions = _concat_or_empty(pos_chunks, np.float32, ndim=3)
    velocities = _concat_or_empty(vel_chunks, np.float32, ndim=3)
    masses = _concat_or_empty(mass_chunks, np.float32)
    previous_accelerations = _concat_or_empty(accel_chunks, np.float32, ndim=3)
    order = np.argsort(ids)
    return (
        ids[order],
        positions[order],
        velocities[order],
        masses[order],
        previous_accelerations[order],
    )


def merge_ghost_payloads(payloads):
    ids_chunks = []
    pos_chunks = []
    vel_chunks = []
    mass_chunks = []
    accel_chunks = []
    for ids, positions, velocities, masses, previous_accelerations in payloads:
        if ids.size == 0:
            continue
        ids_chunks.append(ids)
        pos_chunks.append(positions)
        vel_chunks.append(velocities)
        mass_chunks.append(masses)
        accel_chunks.append(previous_accelerations)
    if not ids_chunks:
        return empty_ghost_payload()
    ids = _concat_or_empty(ids_chunks, np.int64)
    positions = _concat_or_empty(pos_chunks, np.float32, ndim=3)
    velocities = _concat_or_empty(vel_chunks, np.float32, ndim=3)
    masses = _concat_or_empty(mass_chunks, np.float32)
    previous_accelerations = _concat_or_empty(accel_chunks, np.float32, ndim=3)
    order = np.argsort(ids)
    ids = ids[order]
    positions = positions[order]
    velocities = velocities[order]
    masses = masses[order]
    previous_accelerations = previous_accelerations[order]
    if ids.size <= 1:
        return ids, positions, velocities, masses, previous_accelerations
    unique_mask = np.ones(ids.shape[0], dtype=bool)
    unique_mask[1:] = ids[1:] != ids[:-1]
    return (
        ids[unique_mask],
        positions[unique_mask],
        velocities[unique_mask],
        masses[unique_mask],
        previous_accelerations[unique_mask],
    )


def finalize_cell_centers(cell_masses, weighted_positions, cell_com_positions):
    cell_com_positions.fill(0.0)
    non_zero_mask = cell_masses > 0.0
    cell_com_positions[non_zero_mask] = (
        weighted_positions[non_zero_mask] / cell_masses[non_zero_mask, np.newaxis]
    )


@njit
def update_stars_in_grid_global(
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    cell_masses.fill(0.0)
    cell_com_positions.fill(0.0)
    for ibody in range(positions.shape[0]):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = (
            cell_idx[0]
            + cell_idx[1] * n_cells[0]
            + cell_idx[2] * n_cells[0] * n_cells[1]
        )
        mass = masses[ibody]
        cell_masses[linear_idx] += mass
        cell_com_positions[linear_idx] += positions[ibody] * mass
    for index in range(cell_masses.shape[0]):
        if cell_masses[index] > 0.0:
            cell_com_positions[index] /= cell_masses[index]


@njit
def accumulate_global_cells(
    cell_masses: np.ndarray,
    weighted_positions: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    cell_masses.fill(0.0)
    weighted_positions.fill(0.0)
    for ibody in range(positions.shape[0]):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = (
            cell_idx[0]
            + cell_idx[1] * n_cells[0]
            + cell_idx[2] * n_cells[0] * n_cells[1]
        )
        mass = masses[ibody]
        cell_masses[linear_idx] += mass
        weighted_positions[linear_idx] += positions[ibody] * mass


@njit
def update_stars_in_grid_local(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    active_positions: np.ndarray,
    active_masses: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells_global: np.ndarray,
    ghost_cell_ranges: np.ndarray,
    ghost_cell_dims: np.ndarray,
):
    n_active = active_positions.shape[0]
    cell_start_indices.fill(-1)
    cell_masses.fill(0.0)
    cell_com_positions.fill(0.0)
    if n_active == 0:
        cell_start_indices[0] = 0
        return

    cell_counts = np.zeros(np.prod(ghost_cell_dims), dtype=np.int64)
    for ibody in range(n_active):
        cell_idx = np.floor((active_positions[ibody] - grid_min) / cell_size).astype(np.int64)
        inside = True
        for axis in range(3):
            if cell_idx[axis] >= n_cells_global[axis]:
                cell_idx[axis] = n_cells_global[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
            if cell_idx[axis] < ghost_cell_ranges[axis, 0] or cell_idx[axis] >= ghost_cell_ranges[axis, 1]:
                inside = False
                break
        if not inside:
            continue
        local_idx0 = cell_idx[0] - ghost_cell_ranges[0, 0]
        local_idx1 = cell_idx[1] - ghost_cell_ranges[1, 0]
        local_idx2 = cell_idx[2] - ghost_cell_ranges[2, 0]
        linear_idx = (
            local_idx0
            + local_idx1 * ghost_cell_dims[0]
            + local_idx2 * ghost_cell_dims[0] * ghost_cell_dims[1]
        )
        cell_counts[linear_idx] += 1

    running_index = 0
    for index in range(cell_counts.shape[0]):
        cell_start_indices[index] = running_index
        running_index += cell_counts[index]
    cell_start_indices[cell_counts.shape[0]] = running_index

    current_counts = np.zeros(cell_counts.shape[0], dtype=np.int64)
    for ibody in range(n_active):
        cell_idx = np.floor((active_positions[ibody] - grid_min) / cell_size).astype(np.int64)
        inside = True
        for axis in range(3):
            if cell_idx[axis] >= n_cells_global[axis]:
                cell_idx[axis] = n_cells_global[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
            if cell_idx[axis] < ghost_cell_ranges[axis, 0] or cell_idx[axis] >= ghost_cell_ranges[axis, 1]:
                inside = False
                break
        if not inside:
            continue
        local_idx0 = cell_idx[0] - ghost_cell_ranges[0, 0]
        local_idx1 = cell_idx[1] - ghost_cell_ranges[1, 0]
        local_idx2 = cell_idx[2] - ghost_cell_ranges[2, 0]
        linear_idx = (
            local_idx0
            + local_idx1 * ghost_cell_dims[0]
            + local_idx2 * ghost_cell_dims[0] * ghost_cell_dims[1]
        )
        index_in_cell = cell_start_indices[linear_idx] + current_counts[linear_idx]
        body_indices[index_in_cell] = ibody
        current_counts[linear_idx] += 1

    for index in range(cell_counts.shape[0]):
        start_idx = cell_start_indices[index]
        end_idx = cell_start_indices[index + 1]
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        for body_position in range(start_idx, end_idx):
            active_index = body_indices[body_position]
            mass = active_masses[active_index]
            cell_mass += mass
            com_position += active_positions[active_index] * mass
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[index] = cell_mass
        cell_com_positions[index] = com_position


@njit(parallel=True)
def compute_global_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    accelerations = np.zeros_like(positions)
    for ibody in prange(positions.shape[0]):
        position = positions[ibody]
        cell_idx = np.floor((position - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    linear_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    cell_mass = cell_masses[linear_idx]
                    if cell_mass <= 0.0:
                        continue
                    direction = cell_com_positions[linear_idx] - position
                    distance = np.sqrt(
                        direction[0] * direction[0]
                        + direction[1] * direction[1]
                        + direction[2] * direction[2]
                    )
                    if distance > 1.0e-10:
                        inv_dist3 = 1.0 / (distance ** 3)
                        accelerations[ibody] += G * direction * inv_dist3 * cell_mass
    return accelerations


@njit(parallel=True)
def compute_local_accelerations(
    owned_ids: np.ndarray,
    owned_positions: np.ndarray,
    active_ids: np.ndarray,
    active_positions: np.ndarray,
    active_masses: np.ndarray,
    local_cell_start_indices: np.ndarray,
    local_body_indices: np.ndarray,
    local_cell_masses: np.ndarray,
    local_cell_com_positions: np.ndarray,
    global_cell_masses: np.ndarray,
    global_cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    ghost_cell_ranges: np.ndarray,
    ghost_cell_dims: np.ndarray,
):
    accelerations = np.zeros((owned_positions.shape[0], 3), dtype=np.float32)
    for ibody in prange(owned_positions.shape[0]):
        position = owned_positions[ibody]
        global_body_id = owned_ids[ibody]
        cell_idx = np.floor((position - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    linear_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    near_cell = (
                        abs(ix - cell_idx[0]) <= GHOST_LAYER
                        and abs(iy - cell_idx[1]) <= GHOST_LAYER
                        and abs(iz - cell_idx[2]) <= GHOST_LAYER
                    )
                    if near_cell:
                        if (
                            ix >= ghost_cell_ranges[0, 0]
                            and ix < ghost_cell_ranges[0, 1]
                            and iy >= ghost_cell_ranges[1, 0]
                            and iy < ghost_cell_ranges[1, 1]
                            and iz >= ghost_cell_ranges[2, 0]
                            and iz < ghost_cell_ranges[2, 1]
                        ):
                            local_ix = ix - ghost_cell_ranges[0, 0]
                            local_iy = iy - ghost_cell_ranges[1, 0]
                            local_iz = iz - ghost_cell_ranges[2, 0]
                            local_linear_idx = (
                                local_ix
                                + local_iy * ghost_cell_dims[0]
                                + local_iz * ghost_cell_dims[0] * ghost_cell_dims[1]
                            )
                            start_idx = local_cell_start_indices[local_linear_idx]
                            end_idx = local_cell_start_indices[local_linear_idx + 1]
                            for body_position in range(start_idx, end_idx):
                                active_index = local_body_indices[body_position]
                                if active_ids[active_index] == global_body_id:
                                    continue
                                direction = active_positions[active_index] - position
                                distance = np.sqrt(
                                    direction[0] * direction[0]
                                    + direction[1] * direction[1]
                                    + direction[2] * direction[2]
                                )
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    accelerations[ibody] += (
                                        G * direction * inv_dist3 * active_masses[active_index]
                                    )
                        else:
                            cell_mass = global_cell_masses[linear_idx]
                            if cell_mass > 0.0:
                                direction = global_cell_com_positions[linear_idx] - position
                                distance = np.sqrt(
                                    direction[0] * direction[0]
                                    + direction[1] * direction[1]
                                    + direction[2] * direction[2]
                                )
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    accelerations[ibody] += G * direction * inv_dist3 * cell_mass
                    else:
                        cell_mass = global_cell_masses[linear_idx]
                        if cell_mass <= 0.0:
                            continue
                        direction = global_cell_com_positions[linear_idx] - position
                        distance = np.sqrt(
                            direction[0] * direction[0]
                            + direction[1] * direction[1]
                            + direction[2] * direction[2]
                        )
                        if distance > 1.0e-10:
                            inv_dist3 = 1.0 / (distance ** 3)
                            accelerations[ibody] += G * direction * inv_dist3 * cell_mass
    return accelerations


class SerialNBodySystem:
    def __init__(self, filename, ncells_per_dir: tuple[int, int, int] = (10, 10, 10)):
        positions, velocities, masses, box, max_mass, colors = load_dataset(filename)
        self.body_ids = np.arange(positions.shape[0], dtype=np.int64)
        self.positions = positions
        self.velocities = velocities
        self.masses = masses
        self.box = box
        self.max_mass = max_mass
        self.colors = colors
        self.n_cells = np.array(ncells_per_dir, dtype=np.int64)
        self.grid_min = np.array(self.box[0], dtype=np.float32)
        self.grid_max = np.array(self.box[1], dtype=np.float32)
        self.cell_size = (self.grid_max - self.grid_min) / self.n_cells
        n_global_cells = int(np.prod(self.n_cells))
        self.cell_masses = np.zeros(n_global_cells, dtype=np.float32)
        self.cell_com_positions = np.zeros((n_global_cells, 3), dtype=np.float32)
        self.local_cell_ranges = np.array(
            [[0, self.n_cells[axis]] for axis in range(3)],
            dtype=np.int64,
        )
        self.local_cell_dims = self.local_cell_ranges[:, 1] - self.local_cell_ranges[:, 0]
        n_local_cells = int(np.prod(self.local_cell_dims))
        self.local_cell_start_indices = np.full(n_local_cells + 1, -1, dtype=np.int64)
        self.local_body_indices = np.empty(self.positions.shape[0], dtype=np.int64)
        self.local_cell_masses = np.zeros(n_local_cells, dtype=np.float32)
        self.local_cell_com_positions = np.zeros((n_local_cells, 3), dtype=np.float32)
        self.update_global_grid()
        self.build_local_grid()

    def update_global_grid(self):
        update_stars_in_grid_global(
            self.cell_masses,
            self.cell_com_positions,
            self.positions,
            self.masses,
            self.grid_min,
            self.cell_size,
            self.n_cells,
        )

    def build_local_grid(self):
        n_local_cells = int(np.prod(self.local_cell_dims))
        self.local_cell_start_indices = np.full(n_local_cells + 1, -1, dtype=np.int64)
        self.local_body_indices = np.empty(self.positions.shape[0], dtype=np.int64)
        self.local_cell_masses = np.zeros(n_local_cells, dtype=np.float32)
        self.local_cell_com_positions = np.zeros((n_local_cells, 3), dtype=np.float32)
        update_stars_in_grid_local(
            self.local_cell_start_indices,
            self.local_body_indices,
            self.local_cell_masses,
            self.local_cell_com_positions,
            self.positions,
            self.masses,
            self.grid_min,
            self.cell_size,
            self.n_cells,
            self.local_cell_ranges,
            self.local_cell_dims,
        )

    def compute_accelerations(self):
        return compute_local_accelerations(
            self.body_ids,
            self.positions,
            self.body_ids,
            self.positions,
            self.masses,
            self.local_cell_start_indices,
            self.local_body_indices,
            self.local_cell_masses,
            self.local_cell_com_positions,
            self.cell_masses,
            self.cell_com_positions,
            self.grid_min,
            self.cell_size,
            self.n_cells,
            self.local_cell_ranges,
            self.local_cell_dims,
        )

    def step(self, dt):
        accel_pre_start = time.perf_counter()
        acceleration = self.compute_accelerations()
        accel_pre_end = time.perf_counter()
        position_update_start = time.perf_counter()
        self.positions += self.velocities * dt + 0.5 * acceleration * dt * dt
        position_update_end = time.perf_counter()
        global_grid_start = time.perf_counter()
        self.update_global_grid()
        global_grid_end = time.perf_counter()
        local_grid_start = time.perf_counter()
        self.build_local_grid()
        local_grid_end = time.perf_counter()
        accel_post_start = time.perf_counter()
        new_acceleration = self.compute_accelerations()
        accel_post_end = time.perf_counter()
        velocity_update_start = time.perf_counter()
        self.velocities += 0.5 * (acceleration + new_acceleration) * dt
        velocity_update_end = time.perf_counter()
        return {
            "worker_accel_pre_ms": 1000.0 * (accel_pre_end - accel_pre_start),
            "worker_position_update_ms": 1000.0 * (position_update_end - position_update_start),
            "worker_migration_ms": 0.0,
            "worker_global_grid_ms": 1000.0 * (global_grid_end - global_grid_start),
            "worker_ghost_exchange_ms": 0.0,
            "worker_local_grid_ms": 1000.0 * (local_grid_end - local_grid_start),
            "worker_accel_post_ms": 1000.0 * (accel_post_end - accel_post_start),
            "worker_velocity_update_ms": 1000.0 * (velocity_update_end - velocity_update_start),
        }


class WorkerSubdomain:
    def __init__(self, rank, size, nb_cells_per_dim):
        self.rank = rank
        self.size = size
        self.nb_cells_per_dim = tuple(int(value) for value in nb_cells_per_dim)
        self.proc_grid = choose_process_grid(size, self.nb_cells_per_dim)
        self.proc_coords = self._get_proc_coords(rank, self.proc_grid)
        self.local_cell_ranges = np.array(
            [
                self._get_cell_range(self.proc_coords[axis], self.proc_grid[axis], self.nb_cells_per_dim[axis])
                for axis in range(3)
            ],
            dtype=np.int64,
        )
        if np.any(self.local_cell_ranges[:, 0] == self.local_cell_ranges[:, 1]):
            raise ValueError(
                f"El rank de calculo {rank} tiene un subdominio vacio con proc_grid={self.proc_grid}"
            )
        self.ghost_cell_ranges = np.array(
            [
                [
                    max(0, self.local_cell_ranges[axis, 0] - GHOST_LAYER),
                    min(self.nb_cells_per_dim[axis], self.local_cell_ranges[axis, 1] + GHOST_LAYER),
                ]
                for axis in range(3)
            ],
            dtype=np.int64,
        )
        self.ghost_cell_dims = self.ghost_cell_ranges[:, 1] - self.ghost_cell_ranges[:, 0]
        self.owner_coord_by_axis = []
        self.ghost_owner_coords_by_axis = []
        for axis in range(3):
            owners = np.empty(self.nb_cells_per_dim[axis], dtype=np.int32)
            axis_ranges = []
            for coord in range(self.proc_grid[axis]):
                start, end = self._get_cell_range(coord, self.proc_grid[axis], self.nb_cells_per_dim[axis])
                axis_ranges.append((start, end))
                owners[start:end] = coord
            self.owner_coord_by_axis.append(owners)
            ghost_owner_coords = []
            for cell_idx in range(self.nb_cells_per_dim[axis]):
                coords = []
                for coord, (start, end) in enumerate(axis_ranges):
                    ghost_start = max(0, start - GHOST_LAYER)
                    ghost_end = min(self.nb_cells_per_dim[axis], end + GHOST_LAYER)
                    if ghost_start <= cell_idx < ghost_end:
                        coords.append(coord)
                ghost_owner_coords.append(tuple(coords))
            self.ghost_owner_coords_by_axis.append(ghost_owner_coords)

    @staticmethod
    def _get_proc_coords(rank, proc_grid):
        px, py, pz = proc_grid
        z = rank // (px * py)
        y = (rank % (px * py)) // px
        x = rank % px
        return (x, y, z)

    @staticmethod
    def _get_cell_range(coord, proc_count, total_cells):
        base = total_cells // proc_count
        rem = total_cells % proc_count
        start = coord * base + min(coord, rem)
        end = start + base + (1 if coord < rem else 0)
        return (start, end)

    def coords_to_rank(self, coords):
        x, y, z = coords
        px, py, _ = self.proc_grid
        return x + y * px + z * px * py

    def clamp_cell_index(self, position, grid_min, cell_size, n_cells):
        cell_idx = np.floor((position - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        return cell_idx

    def owner_rank_for_cell_idx(self, cell_idx):
        coords = (
            int(self.owner_coord_by_axis[0][cell_idx[0]]),
            int(self.owner_coord_by_axis[1][cell_idx[1]]),
            int(self.owner_coord_by_axis[2][cell_idx[2]]),
        )
        return self.coords_to_rank(coords)

    def owner_rank_for_position(self, position, grid_min, cell_size, n_cells):
        return self.owner_rank_for_cell_idx(self.clamp_cell_index(position, grid_min, cell_size, n_cells))

    def ghost_recipient_ranks_for_cell_idx(self, cell_idx):
        coord_candidates = [
            self.ghost_owner_coords_by_axis[axis][int(cell_idx[axis])]
            for axis in range(3)
        ]
        recipients = set()
        for coords in product(*coord_candidates):
            rank = self.coords_to_rank(coords)
            if rank != self.rank:
                recipients.add(rank)
        return sorted(recipients)


class DistributedWorkerSystem:
    def __init__(self, compute_comm, filename, ncells_per_dir: tuple[int, int, int] = (10, 10, 10)):
        self.compute_comm = compute_comm
        self.rank = compute_comm.Get_rank()
        self.size = compute_comm.Get_size()
        self.n_cells = np.array(ncells_per_dir, dtype=np.int64)
        positions, velocities, masses, box, max_mass, _ = load_dataset(filename)
        self.total_bodies = positions.shape[0]
        self.grid_min = np.array(box[0], dtype=np.float32)
        self.grid_max = np.array(box[1], dtype=np.float32)
        self.cell_size = (self.grid_max - self.grid_min) / self.n_cells
        self.subdomain = WorkerSubdomain(self.rank, self.size, tuple(int(v) for v in self.n_cells))
        local_ids = []
        local_positions = []
        local_velocities = []
        local_masses = []
        for ibody in range(self.total_bodies):
            target_rank = self.subdomain.owner_rank_for_position(
                positions[ibody], self.grid_min, self.cell_size, self.n_cells
            )
            if target_rank == self.rank:
                local_ids.append(ibody)
                local_positions.append(positions[ibody])
                local_velocities.append(velocities[ibody])
                local_masses.append(masses[ibody])
        if local_ids:
            self.owned_ids = np.array(local_ids, dtype=np.int64)
            self.owned_positions = np.array(local_positions, dtype=np.float32)
            self.owned_velocities = np.array(local_velocities, dtype=np.float32)
            self.owned_masses = np.array(local_masses, dtype=np.float32)
            self.owned_prev_accelerations = np.zeros((len(local_ids), 3), dtype=np.float32)
        else:
            (
                self.owned_ids,
                self.owned_positions,
                self.owned_velocities,
                self.owned_masses,
                self.owned_prev_accelerations,
            ) = empty_migration_payload()
        n_global_cells = int(np.prod(self.n_cells))
        self.partial_global_cell_masses = np.zeros(n_global_cells, dtype=np.float32)
        self.partial_global_weighted_positions = np.zeros((n_global_cells, 3), dtype=np.float32)
        self.global_weighted_positions = np.zeros((n_global_cells, 3), dtype=np.float32)
        self.global_cell_masses = np.zeros(n_global_cells, dtype=np.float32)
        self.global_cell_com_positions = np.zeros((n_global_cells, 3), dtype=np.float32)
        (
            self.ghost_ids,
            self.ghost_positions,
            self.ghost_velocities,
            self.ghost_masses,
            self.ghost_prev_accelerations,
        ) = empty_ghost_payload()
        self.stored_ids = np.empty(0, dtype=np.int64)
        self.stored_positions = np.empty((0, 3), dtype=np.float32)
        self.stored_velocities = np.empty((0, 3), dtype=np.float32)
        self.stored_masses = np.empty(0, dtype=np.float32)
        self.stored_prev_accelerations = np.empty((0, 3), dtype=np.float32)
        self.active_ids = np.empty(0, dtype=np.int64)
        self.active_positions = np.empty((0, 3), dtype=np.float32)
        self.active_masses = np.empty(0, dtype=np.float32)
        self.local_cell_start_indices = np.full(np.prod(self.subdomain.ghost_cell_dims) + 1, -1, dtype=np.int64)
        self.local_body_indices = np.empty(0, dtype=np.int64)
        self.local_cell_masses = np.zeros(np.prod(self.subdomain.ghost_cell_dims), dtype=np.float32)
        self.local_cell_com_positions = np.zeros((np.prod(self.subdomain.ghost_cell_dims), 3), dtype=np.float32)
        self.render_positions = None
        self.update_global_grid()
        self.exchange_ghost_particles()
        self.refresh_active_particles()
        self.build_local_grid()

    def update_global_grid(self):
        accumulate_global_cells(
            self.partial_global_cell_masses,
            self.partial_global_weighted_positions,
            self.owned_positions,
            self.owned_masses,
            self.grid_min,
            self.cell_size,
            self.n_cells,
        )
        self.compute_comm.Allreduce(self.partial_global_cell_masses, self.global_cell_masses, op=MPI.SUM)
        self.compute_comm.Allreduce(
            self.partial_global_weighted_positions,
            self.global_weighted_positions,
            op=MPI.SUM,
        )
        finalize_cell_centers(
            self.global_cell_masses,
            self.global_weighted_positions,
            self.global_cell_com_positions,
        )

    def migrate_owned_particles(self):
        send_payloads = [empty_migration_payload() for _ in range(self.size)]
        keep_indices = []
        send_indices_by_rank = [[] for _ in range(self.size)]
        for index in range(self.owned_ids.shape[0]):
            target_rank = self.subdomain.owner_rank_for_position(
                self.owned_positions[index], self.grid_min, self.cell_size, self.n_cells
            )
            if target_rank == self.rank:
                keep_indices.append(index)
            else:
                send_indices_by_rank[target_rank].append(index)

        local_payload = take_particle_payload(
            self.owned_ids,
            self.owned_positions,
            self.owned_velocities,
            self.owned_masses,
            self.owned_prev_accelerations,
            keep_indices,
        )
        for target_rank in range(self.size):
            if target_rank == self.rank:
                continue
            send_payloads[target_rank] = take_particle_payload(
                self.owned_ids,
                self.owned_positions,
                self.owned_velocities,
                self.owned_masses,
                self.owned_prev_accelerations,
                send_indices_by_rank[target_rank],
            )

        received_payloads = self.compute_comm.alltoall(send_payloads)
        (
            self.owned_ids,
            self.owned_positions,
            self.owned_velocities,
            self.owned_masses,
            self.owned_prev_accelerations,
        ) = merge_migration_payloads([local_payload, *received_payloads])

    def exchange_ghost_particles(self):
        send_payloads = [empty_ghost_payload() for _ in range(self.size)]
        send_indices_by_rank = [[] for _ in range(self.size)]
        for index in range(self.owned_ids.shape[0]):
            cell_idx = self.subdomain.clamp_cell_index(
                self.owned_positions[index], self.grid_min, self.cell_size, self.n_cells
            )
            for recipient_rank in self.subdomain.ghost_recipient_ranks_for_cell_idx(cell_idx):
                send_indices_by_rank[recipient_rank].append(index)

        for target_rank in range(self.size):
            if target_rank == self.rank:
                continue
            send_payloads[target_rank] = take_particle_payload(
                self.owned_ids,
                self.owned_positions,
                self.owned_velocities,
                self.owned_masses,
                self.owned_prev_accelerations,
                send_indices_by_rank[target_rank],
            )

        received_payloads = self.compute_comm.alltoall(send_payloads)
        (
            self.ghost_ids,
            self.ghost_positions,
            self.ghost_velocities,
            self.ghost_masses,
            self.ghost_prev_accelerations,
        ) = merge_ghost_payloads(received_payloads)

    def refresh_stored_particles(self):
        if self.ghost_ids.size == 0:
            self.stored_ids = self.owned_ids.copy()
            self.stored_positions = self.owned_positions.copy()
            self.stored_velocities = self.owned_velocities.copy()
            self.stored_masses = self.owned_masses.copy()
            self.stored_prev_accelerations = self.owned_prev_accelerations.copy()
            return
        if self.owned_ids.size == 0:
            self.stored_ids = self.ghost_ids.copy()
            self.stored_positions = self.ghost_positions.copy()
            self.stored_velocities = self.ghost_velocities.copy()
            self.stored_masses = self.ghost_masses.copy()
            self.stored_prev_accelerations = self.ghost_prev_accelerations.copy()
            return
        self.stored_ids = np.concatenate((self.owned_ids, self.ghost_ids))
        self.stored_positions = np.concatenate((self.owned_positions, self.ghost_positions), axis=0)
        self.stored_velocities = np.concatenate((self.owned_velocities, self.ghost_velocities), axis=0)
        self.stored_masses = np.concatenate((self.owned_masses, self.ghost_masses))
        self.stored_prev_accelerations = np.concatenate(
            (self.owned_prev_accelerations, self.ghost_prev_accelerations),
            axis=0,
        )

    def refresh_active_particles(self):
        self.refresh_stored_particles()
        self.active_ids = self.stored_ids.copy()
        self.active_positions = self.stored_positions.copy()
        self.active_masses = self.stored_masses.copy()

    def build_local_grid(self):
        n_active = self.active_positions.shape[0]
        n_local_cells = int(np.prod(self.subdomain.ghost_cell_dims))
        self.local_cell_start_indices = np.full(n_local_cells + 1, -1, dtype=np.int64)
        self.local_body_indices = np.empty(n_active, dtype=np.int64)
        self.local_cell_masses = np.zeros(n_local_cells, dtype=np.float32)
        self.local_cell_com_positions = np.zeros((n_local_cells, 3), dtype=np.float32)
        update_stars_in_grid_local(
            self.local_cell_start_indices,
            self.local_body_indices,
            self.local_cell_masses,
            self.local_cell_com_positions,
            self.active_positions,
            self.active_masses,
            self.grid_min,
            self.cell_size,
            self.n_cells,
            self.subdomain.ghost_cell_ranges,
            self.subdomain.ghost_cell_dims,
        )

    def compute_owned_accelerations(self):
        if self.owned_ids.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return compute_local_accelerations(
            self.owned_ids,
            self.owned_positions,
            self.active_ids,
            self.active_positions,
            self.active_masses,
            self.local_cell_start_indices,
            self.local_body_indices,
            self.local_cell_masses,
            self.local_cell_com_positions,
            self.global_cell_masses,
            self.global_cell_com_positions,
            self.grid_min,
            self.cell_size,
            self.n_cells,
            self.subdomain.ghost_cell_ranges,
            self.subdomain.ghost_cell_dims,
        )

    def step(self, dt):
        accel_pre_start = time.perf_counter()
        acceleration = self.compute_owned_accelerations()
        accel_pre_end = time.perf_counter()
        self.owned_prev_accelerations = acceleration
        position_update_start = time.perf_counter()
        if self.owned_positions.shape[0] > 0:
            self.owned_positions += self.owned_velocities * dt + 0.5 * acceleration * dt * dt
        position_update_end = time.perf_counter()
        migration_start = time.perf_counter()
        self.migrate_owned_particles()
        migration_end = time.perf_counter()
        global_grid_start = time.perf_counter()
        self.update_global_grid()
        global_grid_end = time.perf_counter()
        ghost_exchange_start = time.perf_counter()
        self.exchange_ghost_particles()
        ghost_exchange_mid = time.perf_counter()
        self.refresh_active_particles()
        local_grid_start = time.perf_counter()
        self.build_local_grid()
        local_grid_end = time.perf_counter()
        accel_post_start = time.perf_counter()
        new_acceleration = self.compute_owned_accelerations()
        accel_post_end = time.perf_counter()
        velocity_update_start = time.perf_counter()
        if self.owned_velocities.shape[0] > 0:
            self.owned_velocities += 0.5 * (self.owned_prev_accelerations + new_acceleration) * dt
        velocity_update_end = time.perf_counter()
        self.owned_prev_accelerations = new_acceleration
        ghost_exchange_tail_start = time.perf_counter()
        self.exchange_ghost_particles()
        ghost_exchange_tail_end = time.perf_counter()
        self.refresh_stored_particles()
        return {
            "worker_accel_pre_ms": 1000.0 * (accel_pre_end - accel_pre_start),
            "worker_position_update_ms": 1000.0 * (position_update_end - position_update_start),
            "worker_migration_ms": 1000.0 * (migration_end - migration_start),
            "worker_global_grid_ms": 1000.0 * (global_grid_end - global_grid_start),
            "worker_ghost_exchange_ms": 1000.0
            * ((ghost_exchange_mid - ghost_exchange_start) + (ghost_exchange_tail_end - ghost_exchange_tail_start)),
            "worker_local_grid_ms": 1000.0 * (local_grid_end - local_grid_start),
            "worker_accel_post_ms": 1000.0 * (accel_post_end - accel_post_start),
            "worker_velocity_update_ms": 1000.0 * (velocity_update_end - velocity_update_start),
        }

    def gather_positions_for_root(self):
        if self.rank != 0:
            gatherv_1d_to_root(self.compute_comm, self.owned_ids, MPI.INT64_T)
            gatherv_vec3_to_root(self.compute_comm, self.owned_positions, MPI.FLOAT)
            return None
        gathered_ids = gatherv_1d_to_root(self.compute_comm, self.owned_ids, MPI.INT64_T)
        gathered_positions = gatherv_vec3_to_root(self.compute_comm, self.owned_positions, MPI.FLOAT)
        if self.render_positions is None:
            self.render_positions = np.empty((self.total_bodies, 3), dtype=np.float32)
        self.render_positions[gathered_ids] = gathered_positions
        return self.render_positions


def run_serial_root(
    filename,
    geometry=(800, 600),
    ncells_per_dir: tuple[int, int, int] = (10, 10, 10),
    dt=0.001,
    warmup_frames=5,
    max_frames=None,
    benchmark_csv=None,
    worker_threads=1,
):
    system = SerialNBodySystem(filename, ncells_per_dir=ncells_per_dir)
    positions = system.positions
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    metadata = build_metadata(
        filename,
        dt,
        ncells_per_dir,
        mpi_ranks=1,
        worker_threads=worker_threads,
        root_threads=1,
        cpu_budget_used=1,
    )
    frame_count = 0
    measured_frames = 0
    total_breakdown = zero_worker_breakdown()

    def updater(step_dt: float):
        nonlocal frame_count, measured_frames, total_breakdown
        breakdown = system.step(step_dt)
        frame_count += 1
        if frame_count > warmup_frames:
            measured_frames += 1
            for key in WORKER_BREAKDOWN_KEYS:
                total_breakdown[key] += breakdown[key]
        return system.positions

    visualizer = visualizer3d.Visualizer3D(
        positions,
        system.colors,
        intensity,
        [
            [system.box[0][0], system.box[1][0]],
            [system.box[0][1], system.box[1][1]],
            [system.box[0][2], system.box[1][2]],
        ],
        benchmark_csv=None,
        warmup_frames=warmup_frames,
        max_frames=max_frames,
        metadata=metadata,
    )
    render_stats = visualizer.run(updater=updater, dt=dt)
    avg_root_step_ms = render_stats["avg_updater_ms"]
    avg_breakdown = zero_worker_breakdown()
    if measured_frames > 0:
        for key in WORKER_BREAKDOWN_KEYS:
            avg_breakdown[key] = total_breakdown[key] / measured_frames
    breakdown_summary = summarize_worker_breakdown(
        avg_root_step_ms,
        0.0,
        avg_root_step_ms,
        0.0,
        avg_breakdown,
    )
    row = {
        **metadata,
        "warmup_frames": warmup_frames,
        "measured_frames": render_stats["measured_frames"],
        "avg_render_ms": f"{render_stats['avg_render_ms']:.6f}",
        "avg_root_step_ms": f"{avg_root_step_ms:.6f}",
        "avg_root_command_ms": "0.000000",
        "avg_root_wait_result_ms": "0.000000",
        "avg_root_copy_points_ms": f"{render_stats['avg_copy_points_ms']:.6f}",
        "avg_worker_compute_ms": f"{avg_root_step_ms:.6f}",
        "avg_worker_send_positions_ms": "0.000000",
        "avg_coordination_overhead_ms": "0.000000",
        "avg_update_ms": f"{avg_root_step_ms:.6f}",
        **{key: f"{value:.6f}" for key, value in breakdown_summary.items()},
    }
    write_benchmark_row(benchmark_csv, row)
    print(
        f"measured_frames={render_stats['measured_frames']} "
        f"avg_render_ms={render_stats['avg_render_ms']:.6f} "
        f"avg_root_step_ms={avg_root_step_ms:.6f}"
    )


def run_root(
    comm,
    filename,
    geometry=(800, 600),
    ncells_per_dir: tuple[int, int, int] = (10, 10, 10),
    dt=0.001,
    warmup_frames=5,
    max_frames=None,
    benchmark_csv=None,
    worker_threads=1,
    cpu_budget_used=1,
):
    positions, _, masses, box, max_mass, colors = load_dataset(filename)
    intensity = np.clip(masses / max_mass, 0.5, 1.0)
    metadata = build_metadata(
        filename,
        dt,
        ncells_per_dir,
        mpi_ranks=comm.Get_size(),
        worker_threads=worker_threads,
        root_threads=1,
        cpu_budget_used=cpu_budget_used,
    )
    frame_count = 0
    measured_frames = 0
    total_root_command_time = 0.0
    total_root_wait_result_time = 0.0

    def mpi_updater(step_dt: float):
        nonlocal frame_count, measured_frames, total_root_command_time, total_root_wait_result_time
        command_start = time.perf_counter()
        comm.bcast({"command": "step", "dt": float(step_dt)}, root=ROOT_WORLD_RANK)
        command_end = time.perf_counter()
        wait_result_start = time.perf_counter()
        updated_positions = comm.recv(source=1, tag=1)
        wait_result_end = time.perf_counter()
        frame_count += 1
        if frame_count > warmup_frames:
            measured_frames += 1
            total_root_command_time += 1000.0 * (command_end - command_start)
            total_root_wait_result_time += 1000.0 * (wait_result_end - wait_result_start)
        return updated_positions

    visualizer = visualizer3d.Visualizer3D(
        positions,
        colors,
        intensity,
        [
            [box[0][0], box[1][0]],
            [box[0][1], box[1][1]],
            [box[0][2], box[1][2]],
        ],
        benchmark_csv=None,
        warmup_frames=warmup_frames,
        max_frames=max_frames,
        metadata=metadata,
    )

    render_stats = None
    worker_stats = None
    try:
        render_stats = visualizer.run(updater=mpi_updater, dt=dt)
    finally:
        comm.bcast({"command": "stop"}, root=ROOT_WORLD_RANK)
        worker_stats = comm.recv(source=1, tag=2)

    measured_frames = worker_stats["measured_frames"]
    avg_root_command_ms = 0.0
    avg_root_wait_result_ms = 0.0
    if measured_frames > 0:
        avg_root_command_ms = total_root_command_time / measured_frames
        avg_root_wait_result_ms = total_root_wait_result_time / measured_frames
    avg_root_step_ms = render_stats["avg_updater_ms"]
    avg_root_copy_points_ms = render_stats["avg_copy_points_ms"]
    avg_worker_compute_ms = worker_stats["avg_worker_compute_ms"]
    avg_worker_send_positions_ms = worker_stats["avg_worker_send_positions_ms"]
    avg_breakdown = {
        key: worker_stats[f"avg_{key}"]
        for key in WORKER_BREAKDOWN_KEYS
    }
    avg_coordination_overhead_ms = max(
        0.0,
        avg_root_step_ms - avg_root_command_ms - avg_worker_compute_ms - avg_worker_send_positions_ms,
    )
    breakdown_summary = summarize_worker_breakdown(
        avg_worker_compute_ms,
        avg_worker_send_positions_ms,
        avg_root_step_ms,
        avg_root_wait_result_ms,
        avg_breakdown,
    )
    row = {
        **metadata,
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "avg_render_ms": f"{render_stats['avg_render_ms']:.6f}",
        "avg_root_step_ms": f"{avg_root_step_ms:.6f}",
        "avg_root_command_ms": f"{avg_root_command_ms:.6f}",
        "avg_root_wait_result_ms": f"{avg_root_wait_result_ms:.6f}",
        "avg_root_copy_points_ms": f"{avg_root_copy_points_ms:.6f}",
        "avg_worker_compute_ms": f"{avg_worker_compute_ms:.6f}",
        "avg_worker_send_positions_ms": f"{avg_worker_send_positions_ms:.6f}",
        "avg_coordination_overhead_ms": f"{avg_coordination_overhead_ms:.6f}",
        "avg_update_ms": f"{avg_root_step_ms:.6f}",
        **{key: f"{value:.6f}" for key, value in breakdown_summary.items()},
    }
    write_benchmark_row(benchmark_csv, row)
    print(
        f"measured_frames={measured_frames} "
        f"avg_render_ms={render_stats['avg_render_ms']:.6f} "
        f"avg_root_step_ms={avg_root_step_ms:.6f} "
        f"avg_root_command_ms={avg_root_command_ms:.6f} "
        f"avg_root_wait_result_ms={avg_root_wait_result_ms:.6f} "
        f"avg_worker_compute_ms={avg_worker_compute_ms:.6f} "
        f"avg_worker_send_positions_ms={avg_worker_send_positions_ms:.6f}"
    )


def run_worker_group(world_comm, compute_comm, filename, ncells_per_dir, warmup_frames=5):
    system = DistributedWorkerSystem(compute_comm, filename, ncells_per_dir=ncells_per_dir)
    frame_count = 0
    measured_frames = 0
    total_worker_compute_ms = 0.0
    total_worker_send_ms = 0.0
    total_breakdown = zero_worker_breakdown()

    while True:
        message = world_comm.bcast(None, root=ROOT_WORLD_RANK)
        command = message["command"]
        if command == "stop":
            if compute_comm.Get_rank() == 0:
                avg_worker_compute_ms = 0.0
                avg_worker_send_ms = 0.0
                avg_breakdown = zero_worker_breakdown()
                if measured_frames > 0:
                    avg_worker_compute_ms = total_worker_compute_ms / measured_frames
                    avg_worker_send_ms = total_worker_send_ms / measured_frames
                    for key in WORKER_BREAKDOWN_KEYS:
                        avg_breakdown[key] = total_breakdown[key] / measured_frames
                world_comm.send(
                    {
                        "measured_frames": measured_frames,
                        "avg_worker_compute_ms": avg_worker_compute_ms,
                        "avg_worker_send_positions_ms": avg_worker_send_ms,
                        **{f"avg_{key}": value for key, value in avg_breakdown.items()},
                    },
                    dest=ROOT_WORLD_RANK,
                    tag=2,
                )
            break

        compute_start = time.perf_counter()
        local_breakdown = system.step(float(message["dt"]))
        compute_end = time.perf_counter()
        local_compute_ms = 1000.0 * (compute_end - compute_start)

        send_start = time.perf_counter()
        positions_for_root = system.gather_positions_for_root()
        if compute_comm.Get_rank() == 0:
            world_comm.send(positions_for_root, dest=ROOT_WORLD_RANK, tag=1)
        send_end = time.perf_counter()
        local_send_ms = 1000.0 * (send_end - send_start)

        max_compute_ms = compute_comm.reduce(local_compute_ms, op=MPI.MAX, root=0)
        max_send_ms = compute_comm.reduce(local_send_ms, op=MPI.MAX, root=0)
        local_breakdown_array = np.array(
            [local_breakdown[key] for key in WORKER_BREAKDOWN_KEYS],
            dtype=np.float64,
        )
        max_breakdown_array = None
        if compute_comm.Get_rank() == 0:
            max_breakdown_array = np.empty(len(WORKER_BREAKDOWN_KEYS), dtype=np.float64)
        compute_comm.Reduce(
            [local_breakdown_array, MPI.DOUBLE],
            [max_breakdown_array, MPI.DOUBLE] if compute_comm.Get_rank() == 0 else None,
            op=MPI.MAX,
            root=0,
        )

        frame_count += 1
        if frame_count > warmup_frames and compute_comm.Get_rank() == 0:
            measured_frames += 1
            total_worker_compute_ms += max_compute_ms
            total_worker_send_ms += max_send_ms
            for index, key in enumerate(WORKER_BREAKDOWN_KEYS):
                total_breakdown[key] += float(max_breakdown_array[index])


def run_simulation(
    filename,
    geometry=(800, 600),
    ncells_per_dir: tuple[int, int, int] = (10, 10, 10),
    dt=0.001,
    warmup_frames=5,
    max_frames=None,
    benchmark_csv=None,
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    resource_info = configure_rank_resources(comm)
    worker_threads = resource_info["worker_threads"]
    cpu_budget_used = resource_info["cpu_budget_used"]

    if size == 1:
        if rank == ROOT_WORLD_RANK:
            run_serial_root(
                filename,
                geometry=geometry,
                ncells_per_dir=ncells_per_dir,
                dt=dt,
                warmup_frames=warmup_frames,
                max_frames=max_frames,
                benchmark_csv=benchmark_csv,
                worker_threads=worker_threads,
            )
        return

    color = MPI.UNDEFINED if rank == ROOT_WORLD_RANK else 1
    compute_comm = comm.Split(color=color, key=rank)

    if rank == ROOT_WORLD_RANK:
        run_root(
            comm,
            filename,
            geometry=geometry,
            ncells_per_dir=ncells_per_dir,
            dt=dt,
            warmup_frames=warmup_frames,
            max_frames=max_frames,
            benchmark_csv=benchmark_csv,
            worker_threads=worker_threads,
            cpu_budget_used=cpu_budget_used,
        )
    else:
        run_worker_group(
            comm,
            compute_comm,
            filename,
            ncells_per_dir=ncells_per_dir,
            warmup_frames=warmup_frames,
        )


def build_argument_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.abspath(os.path.join(script_dir, "..", "data", "galaxy_1000"))
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default=default_dataset)
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("nx", nargs="?", type=int, default=20)
    parser.add_argument("ny", nargs="?", type=int, default=20)
    parser.add_argument("nz", nargs="?", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--benchmark-csv", default=None)
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    grid_shape = (args.nx, args.ny, args.nz)
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == ROOT_WORLD_RANK:
        requested_threads = get_requested_worker_threads()
        budget_used = required_cpu_budget(MPI.COMM_WORLD.Get_size(), requested_threads)
        print(
            f"simulation={os.path.abspath(args.filename)} "
            f"dt={args.dt} "
            f"grid={grid_shape} "
            f"root_threads=1 "
            f"worker_threads={requested_threads} "
            f"mpi_ranks={MPI.COMM_WORLD.Get_size()} "
            f"cpu_budget_used={budget_used}"
        )
    run_simulation(
        args.filename,
        ncells_per_dir=grid_shape,
        dt=args.dt,
        warmup_frames=args.warmup_frames,
        max_frames=args.max_frames,
        benchmark_csv=args.benchmark_csv,
    )
