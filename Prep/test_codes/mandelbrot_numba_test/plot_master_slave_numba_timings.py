import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATTERN = re.compile(r"master_slave_numba_(\d+)p\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grafica los tiempos MPI de MasterSlaveNumba.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).with_name("benchmark_csv"),
        help="Directorio que contiene los CSV generados por el benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("master_slave_numba_timings.png"),
        help="Ruta de salida para la grafica PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la grafica ademas de guardarla en disco.",
    )
    return parser.parse_args()


def load_aggregate_rows(input_dir: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for csv_path in sorted(input_dir.glob("master_slave_numba_*p.csv")):
        match = CSV_PATTERN.match(csv_path.name)
        if match is None:
            continue

        process_count = int(match.group(1))
        with csv_path.open("r", encoding="ascii", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            aggregate_row = None
            for row in reader:
                if row["rank"] == "ALL":
                    aggregate_row = row
                    break

        if aggregate_row is None:
            raise ValueError(f"No se encontro la fila ALL en {csv_path}")

        rows.append(
            {
                "processes": process_count,
                "compute": float(aggregate_row["compute_time_seconds"]),
                "communication": float(aggregate_row["communication_time_seconds"]),
                "total": float(aggregate_row["total_measured_time_seconds"]),
            }
        )

    rows.sort(key=lambda row: int(row["processes"]))
    return rows


def plot_timings(rows: list[dict[str, float | int]], output_path: Path, show: bool) -> None:
    if not rows:
        raise ValueError("No se encontraron CSV para graficar.")

    processes = [int(row["processes"]) for row in rows]
    compute = [float(row["compute"]) for row in rows]
    communication = [float(row["communication"]) for row in rows]
    total = [float(row["total"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(processes, compute, width=0.8, label="Calculo", color="#2f6db2")
    ax.bar(
        processes,
        communication,
        width=0.8,
        bottom=compute,
        label="Comunicacion",
        color="#d97925",
    )
    ax.plot(
        processes,
        total,
        marker="o",
        linewidth=2.0,
        color="#1d1d1d",
        label="Total medido",
    )

    ax.set_title("MasterSlaveNumba: tiempo de calculo y comunicacion")
    ax.set_xlabel("Numero de procesos MPI")
    ax.set_ylabel("Tiempo (s)")
    ax.set_xticks(processes)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Grafica guardada en: {output_path}")

    print("Resumen agregado:")
    for row in rows:
        print(
            f"  {int(row['processes'])} procesos -> "
            f"calculo={float(row['compute']):.6f} s, "
            f"comunicacion={float(row['communication']):.6f} s, "
            f"total={float(row['total']):.6f} s"
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_aggregate_rows(args.input_dir)
    plot_timings(rows, args.output, args.show)


if __name__ == "__main__":
    main()
