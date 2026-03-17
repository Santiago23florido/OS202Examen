import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["threads"]))
    return rows


def plot_rows(rows, output_path):
    threads = [int(row["threads"]) for row in rows]
    avg_render_ms = [float(row["avg_render_ms"]) for row in rows]
    avg_root_step_ms = [float(row["avg_root_step_ms"]) for row in rows]
    avg_root_command_ms = [float(row["avg_root_command_ms"]) for row in rows]
    avg_root_wait_result_ms = [float(row["avg_root_wait_result_ms"]) for row in rows]
    avg_root_copy_points_ms = [float(row["avg_root_copy_points_ms"]) for row in rows]
    avg_worker_compute_ms = [float(row["avg_worker_compute_ms"]) for row in rows]
    avg_worker_send_positions_ms = [float(row["avg_worker_send_positions_ms"]) for row in rows]
    avg_coordination_overhead_ms = [float(row["avg_coordination_overhead_ms"]) for row in rows]
    baseline_compute = avg_worker_compute_ms[0]
    speedup = [baseline_compute / value for value in avg_worker_compute_ms]

    figure, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].plot(threads, avg_worker_compute_ms, marker="o", label="avg worker compute")
    axes[0].plot(threads, avg_render_ms, marker="s", label="avg render")
    axes[0].plot(threads, avg_root_step_ms, marker="^", label="avg root step")
    axes[0].plot(threads, avg_worker_send_positions_ms, marker="d", label="avg worker send positions")
    axes[0].set_xlabel("threads")
    axes[0].set_ylabel("time (ms)")
    axes[0].set_title("Average frame times")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(threads, speedup, marker="o", color="tab:green", label="compute speedup")
    axes[1].plot(threads, threads, linestyle="--", color="tab:gray", label="ideal speedup")
    axes[1].set_xlabel("threads")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("Compute speedup")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].bar(threads, avg_render_ms, label="render")
    axes[2].bar(threads, avg_root_command_ms, bottom=avg_render_ms, label="root command")
    bottom_after_command = [render + command for render, command in zip(avg_render_ms, avg_root_command_ms)]
    axes[2].bar(threads, avg_worker_compute_ms, bottom=bottom_after_command, label="worker compute")
    bottom_after_compute = [bottom + compute for bottom, compute in zip(bottom_after_command, avg_worker_compute_ms)]
    axes[2].bar(threads, avg_worker_send_positions_ms, bottom=bottom_after_compute, label="worker send positions")
    bottom_after_send = [bottom + send for bottom, send in zip(bottom_after_compute, avg_worker_send_positions_ms)]
    axes[2].bar(threads, avg_coordination_overhead_ms, bottom=bottom_after_send, label="coordination overhead")
    bottom_after_coordination = [bottom + overhead for bottom, overhead in zip(bottom_after_send, avg_coordination_overhead_ms)]
    axes[2].bar(threads, avg_root_copy_points_ms, bottom=bottom_after_coordination, label="root copy points")
    axes[2].set_xlabel("threads")
    axes[2].set_ylabel("time (ms)")
    axes[2].set_title("Per-frame time breakdown")
    axes[2].grid(True)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"plot={output_path}")


def build_argument_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "benchmark_csv", "mpi_separation_timings.csv")
    default_output = os.path.join(script_dir, "benchmark_csv", "mpi_separation_timings.png")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--output", default=default_output)
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    plot_rows(load_rows(args.csv_path), args.output)
