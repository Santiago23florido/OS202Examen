#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmarks():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "benchmark_csv", "mpi_subdomain_timings.csv")
    output_file = os.path.join(script_dir, "benchmark_csv", "mpi_subdomain_timings.png")

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("No data in CSV file.")
        return

    df["mpi_ranks"] = df["mpi_ranks"].astype(int)
    df = df.sort_values("mpi_ranks")

    if len(df) < 2:
        print(f"Warning: Only {len(df)} data point(s). Need at least 2 for speedup analysis.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("MPI Subdomain Decomposition Performance Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Execution time components
    ax1 = axes[0]
    x_pos = range(len(df))
    components = ["avg_render_ms", "avg_root_command_ms", "avg_root_copy_points_ms"]
    width = 0.6
    bottom = [0] * len(df)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, component in enumerate(components):
        if component in df.columns:
            values = df[component].astype(float).values
            ax1.bar(x_pos, values, width, label=component.replace("avg_", "").replace("_ms", ""), 
                   bottom=bottom, color=colors[i])
            bottom = [bottom[j] + values[j] for j in range(len(bottom))]

    ax1.set_xlabel("MPI Ranks")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Execution Time Components")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df["mpi_ranks"].values)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Speedup
    ax2 = axes[1]
    baseline_time = df["avg_root_command_ms"].iloc[0] if len(df) > 0 else 1.0
    speedups = baseline_time / df["avg_root_command_ms"].astype(float).values
    ax2.plot(df["mpi_ranks"].values, speedups, marker="o", linewidth=2, markersize=8, color="darkgreen")
    ax2.plot(df["mpi_ranks"].values, df["mpi_ranks"].values, "r--", label="Ideal (linear)", linewidth=2)
    ax2.set_xlabel("MPI Ranks")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup vs Ideal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Total time per rank
    ax3 = axes[2]
    total_times = df["avg_root_step_ms"].astype(float).values
    ax3.bar(x_pos, total_times, color="#d62728", width=0.6)
    ax3.set_xlabel("MPI Ranks")
    ax3.set_ylabel("Time (ms)")
    ax3.set_title("Total Step Time")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df["mpi_ranks"].values)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    plot_benchmarks()
