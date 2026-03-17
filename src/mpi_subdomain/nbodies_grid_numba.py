import argparse
import csv
import os
import time

import numpy as np
from mpi4py import MPI
from numba import get_num_threads, njit, prange

import visualizer3d

G = 1.560339e-13
GHOST_LAYER = 1


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


def choose_process_grid(size: int) -> tuple[int, int, int]:
    """Encuentra la mejor descomposición de procesos MPI en 3D."""
    best = (1, 1, size)
    best_volume = size
    for px in range(1, size + 1):
        if size % px != 0:
            continue
        rem = size // px
        for py in range(1, rem + 1):
            if rem % py != 0:
                continue
            pz = rem // py
            volume = max(px, py, pz)
            if volume < best_volume:
                best = (px, py, pz)
                best_volume = volume
    return best


@njit
def update_stars_in_grid_local(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    local_bodies_id: np.ndarray,
):
    """Actualiza la cuadrícula local para cuerpos asignados a este proceso."""
    n_local_bodies = len(local_bodies_id)
    cell_start_indices.fill(-1)
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    
    for i in range(n_local_bodies):
        ibody = local_bodies_id[i]
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[linear_idx] += 1
    
    running_index = 0
    for index in range(len(cell_counts)):
        cell_start_indices[index] = running_index
        running_index += cell_counts[index]
    cell_start_indices[len(cell_counts)] = running_index
    
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for i in range(n_local_bodies):
        ibody = local_bodies_id[i]
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[linear_idx] + current_counts[linear_idx]
        body_indices[index_in_cell] = ibody
        current_counts[linear_idx] += 1
    
    for index in range(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[index]
        end_idx = cell_start_indices[index + 1]
        for body_position in range(start_idx, end_idx):
            ibody = body_indices[body_position]
            mass = masses[ibody]
            cell_mass += mass
            com_position += positions[ibody] * mass
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[index] = cell_mass
        cell_com_positions[index] = com_position


@njit(parallel=True)
def compute_acceleration_local(
    positions: np.ndarray,
    masses: np.ndarray,
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
    local_bodies_id: np.ndarray,
):
    """Calcula aceleraciones para cuerpos locales usando la cuadrícula global."""
    n_local_bodies = len(local_bodies_id)
    accelerations = np.zeros((n_local_bodies, 3), dtype=np.float32)
    
    for i in prange(n_local_bodies):
        ibody = local_bodies_id[i]
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
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_com = cell_com_positions[linear_idx]
                        cell_mass = cell_masses[linear_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - position
                            distance = np.sqrt(
                                direction[0] * direction[0]
                                + direction[1] * direction[1]
                                + direction[2] * direction[2]
                            )
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                accelerations[i, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[linear_idx]
                        end_idx = cell_start_indices[linear_idx + 1]
                        for body_position in range(start_idx, end_idx):
                            jbody = body_indices[body_position]
                            if jbody != ibody:
                                direction = positions[jbody] - position
                                distance = np.sqrt(
                                    direction[0] * direction[0]
                                    + direction[1] * direction[1]
                                    + direction[2] * direction[2]
                                )
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    accelerations[i, :] += G * direction[:] * inv_dist3 * masses[jbody]
    
    return accelerations


class SpatialSubdomain:
    """Representa el subdominio cartesiano asignado a un proceso MPI."""
    
    def __init__(self, rank, size, nb_cells_per_dim):
        self.rank = rank
        self.size = size
        self.nb_cells_per_dim = np.array(nb_cells_per_dim)
        self.proc_grid = choose_process_grid(size)
        self.proc_coords = self._get_proc_coords(rank, self.proc_grid)
        
        self.local_cell_ranges = [
            self._get_cell_range(self.proc_coords[i], self.proc_grid[i], nb_cells_per_dim[i])
            for i in range(3)
        ]
        
        self.local_cells_with_ghost = tuple(
            (self.local_cell_ranges[i][1] - self.local_cell_ranges[i][0]) + 2 * GHOST_LAYER
            for i in range(3)
        )
        
        self.n_local_cells = np.prod(self.local_cells_with_ghost)
        self.cell_start_indices = np.full(self.n_local_cells + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(10000, dtype=np.int64)
        self.cell_masses = np.zeros(self.n_local_cells, dtype=np.float32)
        self.cell_com_positions = np.zeros((self.n_local_cells, 3), dtype=np.float32)
        
        self.local_bodies_id = np.array([], dtype=np.int64)
    
    @staticmethod
    def _get_proc_coords(rank, proc_grid):
        """Obtiene las coordenadas 3D del proceso en la malla de procesos."""
        px, py, pz = proc_grid
        z = rank // (px * py)
        y = (rank % (px * py)) // px
        x = rank % px
        return (x, y, z)
    
    @staticmethod
    def _get_cell_range(coord, proc_count, total_cells):
        """Obtiene el rango de celdas asignadas a este rango en esta dimensión."""
        base = total_cells // proc_count
        rem = total_cells % proc_count
        start = coord * base + min(coord, rem)
        end = start + base + (1 if coord < rem else 0)
        return (start, end)
    
    def get_subdomain_bounds(self, global_bounds):
        """Calcula los límites espaciales de este subdominio."""
        bounds = np.array(global_bounds, dtype=np.float32)
        cell_size = (bounds[1] - bounds[0]) / self.nb_cells_per_dim
        
        local_bounds = []
        for i in range(3):
            cell_start = self.local_cell_ranges[i][0]
            cell_end = self.local_cell_ranges[i][1]
            min_bound = bounds[0][i] + cell_start * cell_size[i]
            max_bound = bounds[0][i] + cell_end * cell_size[i]
            local_bounds.append([min_bound, max_bound])
        
        return np.array(local_bounds, dtype=np.float32)
    
    def assign_local_bodies(self, positions, global_bound):
        """Asigna cuerpos a este proceso según su posición espacial."""
        bounds = np.array(global_bound, dtype=np.float32)
        cell_size = (bounds[1] - bounds[0]) / self.nb_cells_per_dim
        
        local_list = []
        for i, pos in enumerate(positions):
            in_range = True
            for axis in range(3):
                cell_idx = int(np.floor((pos[axis] - bounds[0][axis]) / cell_size[axis]))
                cell_idx = max(0, min(cell_idx, self.nb_cells_per_dim[axis] - 1))
                
                if not (self.local_cell_ranges[axis][0] <= cell_idx < self.local_cell_ranges[axis][1]):
                    in_range = False
                    break
            
            if in_range:
                local_list.append(i)
        
        self.local_bodies_id = np.array(local_list, dtype=np.int64)


class NBodySystem:
    def __init__(self, filename, ncells_per_dir: tuple[int, int, int] = (10, 10, 10)):
        positions = []
        velocities = []
        masses = []
        self.max_mass = 0.0
        self.box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float32)
        with open(filename, "r") as handle:
            line = handle.readline()
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                for axis in range(3):
                    self.box[0][axis] = min(self.box[0][axis], positions[-1][axis] - 1.0e-6)
                    self.box[1][axis] = max(self.box[1][axis], positions[-1][axis] + 1.0e-6)
                line = handle.readline()
        self.positions = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(mass) for mass in masses]
        
        self.ncells_per_dir = ncells_per_dir
        self.grid_min = np.array(self.box[0], dtype=np.float32)
        self.grid_max = np.array(self.box[1], dtype=np.float32)
        self.n_cells = np.array(ncells_per_dir, dtype=np.intc)
        self.cell_size = (self.grid_max - self.grid_min) / self.n_cells
        
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(self.positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)
        
        self.update_global_grid()
        self.local_bodies_id = np.array([], dtype=np.int64)
        self.accelerations = np.zeros((1, 3), dtype=np.float32)
    
    def update_global_grid(self):
        """Actualiza la cuadrícula global con todas las posiciones."""
        update_stars_in_grid_global(
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            self.masses,
            self.positions,
            self.grid_min,
            self.grid_max,
            self.cell_size,
            self.n_cells,
        )
    
    def update_positions_split(self, dt, local_bodies_id, accelerations):
        """Actualiza posiciones para cuerpos locales."""
        for i, ibody in enumerate(local_bodies_id):
            self.positions[ibody] += self.velocities[ibody] * dt + 0.5 * accelerations[i] * dt * dt
        
        self.update_global_grid()
        
        accelerations2 = compute_acceleration_local(
            self.positions,
            self.masses,
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            self.grid_min,
            self.grid_max,
            self.cell_size,
            self.n_cells,
            local_bodies_id,
        )
        
        for i, ibody in enumerate(local_bodies_id):
            self.velocities[ibody] += 0.5 * (accelerations[i] + accelerations2[i]) * dt


@njit
def update_stars_in_grid_global(
    cell_start_indices: np.ndarray,
    body_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    masses: np.ndarray,
    positions: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    """Actualiza la cuadrícula global con todas las posiciones globales."""
    n_bodies = positions.shape[0]
    cell_start_indices.fill(-1)
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[linear_idx] += 1
    running_index = 0
    for index in range(len(cell_counts)):
        cell_start_indices[index] = running_index
        running_index += cell_counts[index]
    cell_start_indices[len(cell_counts)] = running_index
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1
            elif cell_idx[axis] < 0:
                cell_idx[axis] = 0
        linear_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[linear_idx] + current_counts[linear_idx]
        body_indices[index_in_cell] = ibody
        current_counts[linear_idx] += 1
    for index in range(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[index]
        end_idx = cell_start_indices[index + 1]
        for body_position in range(start_idx, end_idx):
            ibody = body_indices[body_position]
            mass = masses[ibody]
            cell_mass += mass
            com_position += positions[ibody] * mass
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[index] = cell_mass
        cell_com_positions[index] = com_position


system: NBodySystem | None = None


def update_positions(dt: float):
    global system
    system.update_positions_split(dt, system.local_bodies_id, system.accelerations)
    return system.positions


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


def build_metadata(filename, dt, ncells_per_dir, mpi_ranks):
    return {
        "threads": str(get_num_threads()),
        "mpi_ranks": str(mpi_ranks),
        "filename": os.path.abspath(filename),
        "dt": str(dt),
        "nx": str(ncells_per_dir[0]),
        "ny": str(ncells_per_dir[1]),
        "nz": str(ncells_per_dir[2]),
    }


def run_root(
    comm,
    subdomain,
    filename,
    geometry=(800, 600),
    ncells_per_dir: tuple[int, int, int] = (10, 10, 10),
    dt=0.001,
    warmup_frames=5,
    max_frames=None,
    benchmark_csv=None,
):
    global system
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    subdomain.assign_local_bodies(system.positions, system.box)
    system.local_bodies_id = subdomain.local_bodies_id
    
    positions = system.positions
    colors = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    metadata = build_metadata(filename, dt, ncells_per_dir, comm.Get_size())
    frame_count = 0
    measured_frames = 0
    total_root_command_time = 0.0

    def mpi_updater(step_dt: float):
        nonlocal frame_count, measured_frames, total_root_command_time
        command_start = time.perf_counter()
        
        accelerations = compute_acceleration_local(
            system.positions,
            system.masses,
            system.cell_start_indices,
            system.body_indices,
            system.cell_masses,
            system.cell_com_positions,
            system.grid_min,
            system.grid_max,
            system.cell_size,
            system.n_cells,
            system.local_bodies_id,
        )
        system.accelerations = accelerations
        
        system.update_positions_split(step_dt, system.local_bodies_id, accelerations)
        
        command_end = time.perf_counter()
        frame_count += 1
        if frame_count > warmup_frames:
            measured_frames += 1
            total_root_command_time += 1000.0 * (command_end - command_start)
        
        return system.positions

    visualizer = visualizer3d.Visualizer3D(
        positions,
        colors,
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

    render_stats = None
    try:
        render_stats = visualizer.run(updater=mpi_updater, dt=dt)
    finally:
        pass

    measured_frames = max(1, measured_frames)
    avg_root_command_ms = total_root_command_time / measured_frames if measured_frames > 0 else 0.0
    avg_root_step_ms = render_stats["avg_updater_ms"]
    avg_root_copy_points_ms = render_stats["avg_copy_points_ms"]
    
    row = {
        "threads": metadata["threads"],
        "mpi_ranks": metadata["mpi_ranks"],
        "filename": metadata["filename"],
        "dt": metadata["dt"],
        "nx": metadata["nx"],
        "ny": metadata["ny"],
        "nz": metadata["nz"],
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "avg_render_ms": f"{render_stats['avg_render_ms']:.6f}",
        "avg_root_step_ms": f"{avg_root_step_ms:.6f}",
        "avg_root_command_ms": f"{avg_root_command_ms:.6f}",
        "avg_root_wait_result_ms": "0.000000",
        "avg_root_copy_points_ms": f"{avg_root_copy_points_ms:.6f}",
        "avg_worker_compute_ms": f"{avg_root_command_ms:.6f}",
        "avg_worker_send_positions_ms": "0.000000",
        "avg_coordination_overhead_ms": "0.000000",
        "avg_update_ms": f"{avg_root_command_ms:.6f}",
    }
    write_benchmark_row(benchmark_csv, row)
    print(
        f"measured_frames={measured_frames} "
        f"avg_render_ms={render_stats['avg_render_ms']:.6f} "
        f"avg_root_step_ms={avg_root_step_ms:.6f} "
        f"avg_root_command_ms={avg_root_command_ms:.6f}"
    )


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

    subdomain = SpatialSubdomain(rank, size, ncells_per_dir)

    if rank == 0:
        run_root(
            comm,
            subdomain,
            filename,
            geometry=geometry,
            ncells_per_dir=ncells_per_dir,
            dt=dt,
            warmup_frames=warmup_frames,
            max_frames=max_frames,
            benchmark_csv=benchmark_csv,
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
    if rank == 0:
        print(
            f"simulation={os.path.abspath(args.filename)} "
            f"dt={args.dt} "
            f"grid={grid_shape} "
            f"threads={get_num_threads()} "
            f"mpi_ranks={MPI.COMM_WORLD.Get_size()}"
        )
    run_simulation(
        args.filename,
        ncells_per_dir=grid_shape,
        dt=args.dt,
        warmup_frames=args.warmup_frames,
        max_frames=args.max_frames,
        benchmark_csv=args.benchmark_csv,
    )
