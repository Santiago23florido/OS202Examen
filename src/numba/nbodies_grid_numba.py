import argparse
import os

import numpy as np
import visualizer3d
from numba import get_num_threads, njit, prange

G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


@njit(parallel=True)
def update_stars_in_grid(
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
    for index in prange(len(cell_counts)):
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
def compute_acceleration(
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
):
    n_bodies = positions.shape[0]
    accelerations = np.zeros_like(positions)
    for ibody in prange(n_bodies):
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
                                accelerations[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
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
                                    accelerations[ibody, :] += G * direction[:] * inv_dist3 * masses[jbody]
    return accelerations


class SpatialGrid:
    def __init__(self, positions: np.ndarray, nb_cells_per_dim: tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)

    def update_bounds(self, positions: np.ndarray):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells

    def update(self, positions: np.ndarray, masses: np.ndarray):
        update_stars_in_grid(
            self.cell_start_indices,
            self.body_indices,
            self.cell_masses,
            self.cell_com_positions,
            masses,
            positions,
            self.min_bounds,
            self.max_bounds,
            self.cell_size,
            self.n_cells,
        )


class NBodySystem:
    def __init__(self, filename, ncells_per_dir: tuple[int, int, int] = (10, 10, 10)):
        positions = []
        velocities = []
        masses = []
        self.max_mass = 0.0
        self.box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
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
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)

    def update_positions(self, dt):
        acceleration = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.max_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.positions += self.velocities * dt + 0.5 * acceleration * dt * dt
        self.grid.update(self.positions, self.masses)
        new_acceleration = compute_acceleration(
            self.positions,
            self.masses,
            self.grid.cell_start_indices,
            self.grid.body_indices,
            self.grid.cell_masses,
            self.grid.cell_com_positions,
            self.grid.min_bounds,
            self.grid.max_bounds,
            self.grid.cell_size,
            self.grid.n_cells,
        )
        self.velocities += 0.5 * (acceleration + new_acceleration) * dt


system: NBodySystem | None = None


def update_positions(dt: float):
    global system
    system.update_positions(dt)
    return system.positions


def run_simulation(
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
    positions = system.positions
    colors = system.colors
    intensity = np.clip(system.masses / system.max_mass, 0.5, 1.0)
    metadata = {
        "threads": str(get_num_threads()),
        "filename": os.path.abspath(filename),
        "dt": str(dt),
        "nx": str(ncells_per_dir[0]),
        "ny": str(ncells_per_dir[1]),
        "nz": str(ncells_per_dir[2]),
    }
    visualizer = visualizer3d.Visualizer3D(
        positions,
        colors,
        intensity,
        [
            [system.box[0][0], system.box[1][0]],
            [system.box[0][1], system.box[1][1]],
            [system.box[0][2], system.box[1][2]],
        ],
        benchmark_csv=benchmark_csv,
        warmup_frames=warmup_frames,
        max_frames=max_frames,
        metadata=metadata,
    )
    visualizer.run(updater=update_positions, dt=dt)


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
    print(
        f"simulation={os.path.abspath(args.filename)} "
        f"dt={args.dt} "
        f"grid={grid_shape} "
        f"threads={get_num_threads()}"
    )
    run_simulation(
        args.filename,
        ncells_per_dir=grid_shape,
        dt=args.dt,
        warmup_frames=args.warmup_frames,
        max_frames=args.max_frames,
        benchmark_csv=args.benchmark_csv,
    )
