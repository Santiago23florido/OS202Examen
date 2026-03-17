#!/usr/bin/env python3

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


BREAKDOWN_COLUMNS = [
    "avg_worker_accel_pre_ms",
    "avg_worker_position_update_ms",
    "avg_worker_migration_ms",
    "avg_worker_global_grid_ms",
    "avg_worker_ghost_exchange_ms",
    "avg_worker_local_grid_ms",
    "avg_worker_accel_post_ms",
    "avg_worker_velocity_update_ms",
    "avg_worker_force_ms",
    "avg_worker_integration_ms",
    "avg_worker_sync_ms",
    "avg_worker_step_accounted_ms",
    "avg_worker_unaccounted_ms",
    "pct_worker_force_of_compute",
    "pct_worker_sync_of_compute",
    "pct_worker_send_of_step",
    "pct_root_wait_of_step",
]


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    numeric_columns = [
        "threads",
        "worker_threads",
        "root_threads",
        "worker_ranks",
        "mpi_ranks",
        "cpu_budget_used",
        "avg_render_ms",
        "avg_root_step_ms",
        "avg_root_command_ms",
        "avg_root_wait_result_ms",
        "avg_root_copy_points_ms",
        "avg_worker_compute_ms",
        "avg_worker_send_positions_ms",
        "avg_coordination_overhead_ms",
        "avg_update_ms",
        *BREAKDOWN_COLUMNS,
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "worker_threads" not in df.columns and "threads" in df.columns:
        df["worker_threads"] = df["threads"]
    if "root_threads" not in df.columns:
        df["root_threads"] = 1
    if "worker_ranks" not in df.columns and "mpi_ranks" in df.columns:
        df["worker_ranks"] = (df["mpi_ranks"] - 1).clip(lower=0)
    if "cpu_budget_used" not in df.columns and {"mpi_ranks", "worker_threads"} <= set(df.columns):
        df["cpu_budget_used"] = 1 + (df["mpi_ranks"] - 1) * df["worker_threads"]

    return df.sort_values(["worker_threads", "mpi_ranks"])


def filter_dataframe(df, worker_threads=None, mpi_min=None, mpi_max=None):
    filtered = df.copy()
    if worker_threads is not None:
        filtered = filtered[filtered["worker_threads"] == worker_threads]
    if mpi_min is not None:
        filtered = filtered[filtered["mpi_ranks"] >= mpi_min]
    if mpi_max is not None:
        filtered = filtered[filtered["mpi_ranks"] <= mpi_max]
    return filtered.sort_values(["worker_threads", "mpi_ranks"])


def set_integer_rank_ticks(axis, mpi_ranks):
    axis.set_xticks(mpi_ranks)
    axis.set_xticklabels([str(int(rank)) for rank in mpi_ranks])


def plot_time_and_speedup(df_group, output_path):
    worker_threads = int(df_group["worker_threads"].iloc[0])
    mpi_ranks = df_group["mpi_ranks"].astype(int).to_numpy()
    worker_ranks = df_group["worker_ranks"].astype(int).to_numpy()
    avg_render_ms = df_group["avg_render_ms"].astype(float).to_numpy()
    avg_update_ms = df_group["avg_update_ms"].astype(float).to_numpy()
    avg_worker_compute_ms = df_group["avg_worker_compute_ms"].astype(float).to_numpy()

    baseline_update = avg_update_ms[0]
    speedup = baseline_update / avg_update_ms if baseline_update > 0.0 else avg_update_ms
    effective_compute_units = worker_ranks.copy()
    effective_compute_units[effective_compute_units < 1] = 1
    ideal_speedup = effective_compute_units / effective_compute_units[0]
    has_breakdown = set(BREAKDOWN_COLUMNS).issubset(df_group.columns)

    if has_breakdown:
        figure, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.reshape(2, 2)
    else:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes = axes.reshape(1, 2)
    figure.suptitle(
        (
            f"MPI Subdomain | worker_threads={worker_threads} | "
            f"mpi_ranks={mpi_ranks[0]}..{mpi_ranks[-1]} | "
            f"worker_ranks={worker_ranks[0]}..{worker_ranks[-1]}"
        ),
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].plot(mpi_ranks, avg_update_ms, marker="o", linewidth=2, label="avg update")
    axes[0, 0].plot(mpi_ranks, avg_render_ms, marker="s", linewidth=2, label="avg render")
    axes[0, 0].plot(mpi_ranks, avg_worker_compute_ms, marker="^", linewidth=2, label="avg worker compute")
    axes[0, 0].set_xlabel("MPI ranks")
    axes[0, 0].set_ylabel("time (ms)")
    axes[0, 0].set_title("Frame Times")
    set_integer_rank_ticks(axes[0, 0], mpi_ranks)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(mpi_ranks, speedup, marker="o", linewidth=2, color="tab:green", label="speedup")
    axes[0, 1].plot(
        mpi_ranks,
        ideal_speedup,
        linestyle="--",
        linewidth=2,
        color="tab:gray",
        label="ideal from baseline",
    )
    axes[0, 1].set_xlabel("MPI ranks")
    axes[0, 1].set_ylabel("speedup")
    axes[0, 1].set_title("Speedup")
    set_integer_rank_ticks(axes[0, 1], mpi_ranks)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    if has_breakdown:
        force_ms = df_group["avg_worker_force_ms"].astype(float).to_numpy()
        sync_ms = df_group["avg_worker_sync_ms"].astype(float).to_numpy()
        integration_ms = df_group["avg_worker_integration_ms"].astype(float).to_numpy()
        local_grid_ms = df_group["avg_worker_local_grid_ms"].astype(float).to_numpy()
        unaccounted_ms = df_group["avg_worker_unaccounted_ms"].astype(float).to_numpy()
        send_ms = df_group["avg_worker_send_positions_ms"].astype(float).to_numpy()
        pct_force = df_group["pct_worker_force_of_compute"].astype(float).to_numpy()
        pct_sync = df_group["pct_worker_sync_of_compute"].astype(float).to_numpy()
        pct_send = df_group["pct_worker_send_of_step"].astype(float).to_numpy()
        pct_root_wait = df_group["pct_root_wait_of_step"].astype(float).to_numpy()

        axes[1, 0].bar(mpi_ranks, force_ms, label="force", color="#2C7FB8")
        axes[1, 0].bar(mpi_ranks, sync_ms, bottom=force_ms, label="sync", color="#F28E2B")
        bottom_after_sync = force_ms + sync_ms
        axes[1, 0].bar(
            mpi_ranks,
            integration_ms,
            bottom=bottom_after_sync,
            label="integration",
            color="#59A14F",
        )
        bottom_after_integration = bottom_after_sync + integration_ms
        axes[1, 0].bar(
            mpi_ranks,
            local_grid_ms,
            bottom=bottom_after_integration,
            label="local grid",
            color="#9C755F",
        )
        bottom_after_local_grid = bottom_after_integration + local_grid_ms
        axes[1, 0].bar(
            mpi_ranks,
            unaccounted_ms,
            bottom=bottom_after_local_grid,
            label="other",
            color="#BAB0AC",
        )
        axes[1, 0].plot(mpi_ranks, send_ms, marker="o", linewidth=2, color="#D62728", label="send to root")
        axes[1, 0].set_xlabel("MPI ranks")
        axes[1, 0].set_ylabel("time (ms)")
        axes[1, 0].set_title("Worker Cost Breakdown")
        set_integer_rank_ticks(axes[1, 0], mpi_ranks)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].plot(mpi_ranks, pct_force, marker="o", linewidth=2, label="% force / compute")
        axes[1, 1].plot(mpi_ranks, pct_sync, marker="s", linewidth=2, label="% sync / compute")
        axes[1, 1].plot(mpi_ranks, pct_send, marker="^", linewidth=2, label="% send / step")
        axes[1, 1].plot(mpi_ranks, pct_root_wait, marker="d", linewidth=2, label="% root wait / step")
        axes[1, 1].set_xlabel("MPI ranks")
        axes[1, 1].set_ylabel("percentage (%)")
        axes[1, 1].set_title("MPI Cost Ratios")
        set_integer_rank_ticks(axes[1, 1], mpi_ranks)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_argument_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "benchmark_csv", "mpi_subdomain_timings.csv")
    default_output = os.path.join(script_dir, "benchmark_csv", "mpi_subdomain_timings.png")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=default_csv)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--worker-threads", type=int, default=2)
    parser.add_argument("--mpi-min", type=int, default=1)
    parser.add_argument("--mpi-max", type=int, default=7)
    return parser


def main():
    args = build_argument_parser().parse_args()
    if not os.path.exists(args.csv_path):
        print(f"CSV file not found: {args.csv_path}")
        return

    df = load_dataframe(args.csv_path)
    if df.empty:
        print("No data in CSV file.")
        return

    filtered = filter_dataframe(
        df,
        worker_threads=args.worker_threads,
        mpi_min=args.mpi_min,
        mpi_max=args.mpi_max,
    )
    if filtered.empty:
        print("No rows match the requested filters.")
        return

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    plot_time_and_speedup(filtered, args.output)
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
