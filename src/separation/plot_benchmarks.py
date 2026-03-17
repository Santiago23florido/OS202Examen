import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["threads"]))
    return rows


def plot_rows(rows, output_path):
    threads = [int(row["threads"]) for row in rows]
    avg_render_ms = [float(row["avg_render_ms"]) for row in rows]
    avg_update_ms = [float(row["avg_update_ms"]) for row in rows]
    baseline_update = avg_update_ms[0]
    speedup = [baseline_update / value for value in avg_update_ms]

    figure, axes = plt.subplots(2, 1, figsize=(9, 10))

    axes[0].plot(threads, avg_update_ms, marker="o", label="avg update")
    axes[0].plot(threads, avg_render_ms, marker="s", label="avg render")
    axes[0].set_xlabel("threads")
    axes[0].set_ylabel("time (ms)")
    axes[0].set_title("Average frame times")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(threads, speedup, marker="o", color="tab:green", label="update speedup")
    axes[1].plot(threads, threads, linestyle="--", color="tab:gray", label="ideal speedup")
    axes[1].set_xlabel("threads")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("Update speedup")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"plot={output_path}")


def build_argument_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "benchmark_csv", "numba_parallel_timings.csv")
    default_output = os.path.join(script_dir, "benchmark_csv", "numba_parallel_timings.png")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--output", default=default_output)
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    plot_rows(load_rows(args.csv_path), args.output)
