# Ce programme charge des images extraites d'une video, applique le filtre
# d'agrandissement + nettete, puis sauvegarde les images dans un dossier.

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from mpi4py import MPI
from scipy import signal

BATCH_SIZE = 4
WORK_TAG = 100
RESULT_META_TAG = 101
STOP_TAG = 103
ROOT = 0
DEFAULT_INPUT_DIR = Path("datas/perroquets")
DEFAULT_OUTPUT_DIR = Path("sorties/perroquets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtrage MPI d'images extraites d'une video."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Dossier contenant les images a traiter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sauvegarde des images filtrees.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Nombre d'images envoyees dans chaque lot MPI.",
    )
    return parser.parse_args()


def apply_filter(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    img = img.convert("HSV")
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double) / 255.0

    gaussian_mask = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.double,
    ) / 16.0
    blur_image = np.zeros_like(img, dtype=np.double)
    for channel in range(3):
        blur_image[:, :, channel] = signal.convolve2d(
            img[:, :, channel],
            gaussian_mask,
            mode="same",
        )

    sharpen_mask = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.double,
    )
    sharpen_image = np.zeros_like(img)
    sharpen_image[:, :, :2] = blur_image[:, :, :2]
    sharpen_image[:, :, 2] = np.clip(
        signal.convolve2d(blur_image[:, :, 2], sharpen_mask, mode="same"),
        0.0,
        1.0,
    )

    sharpen_image = (sharpen_image * 255.0).astype(np.uint8)
    return Image.fromarray(sharpen_image, "HSV").convert("RGB")


def list_input_images(input_dir: Path) -> list[str]:
    image_names = sorted(path.name for path in input_dir.glob("Perroquet*.jpg"))
    if not image_names:
        image_names = sorted(path.name for path in input_dir.glob("*.jpg"))
    return image_names


def process_batch(
    image_names: list[str],
    input_dir: Path,
    output_dir: Path,
    start_index: int,
    image_count: int,
) -> int:
    end_index = start_index + image_count
    for image_name in image_names[start_index:end_index]:
        output_image = apply_filter(input_dir / image_name)
        output_image.save(output_dir / image_name)
    return image_count


def dispatch_work(
    comm,
    worker: int,
    next_index: int,
    total_images: int,
    batch_size: int,
) -> tuple[int, bool]:
    if next_index >= total_images:
        stop_payload = np.array([-1, 0], dtype=np.intc)
        comm.Send([stop_payload, MPI.INT], dest=worker, tag=STOP_TAG)
        return next_index, False

    image_count = min(batch_size, total_images - next_index)
    payload = np.array([next_index, image_count], dtype=np.intc)
    comm.Send([payload, MPI.INT], dest=worker, tag=WORK_TAG)
    return next_index + image_count, True


def master(
    comm,
    size: int,
    total_images: int,
    batch_size: int,
) -> int:
    next_index = 0
    active_workers = 0
    processed_images = 0

    for worker_rank in range(1, size):
        next_index, has_work = dispatch_work(
            comm,
            worker_rank,
            next_index,
            total_images,
            batch_size,
        )
        if has_work:
            active_workers += 1

    while active_workers > 0:
        status = MPI.Status()
        result_meta = np.empty(2, dtype=np.intc)
        comm.Recv(
            [result_meta, MPI.INT],
            source=MPI.ANY_SOURCE,
            tag=RESULT_META_TAG,
            status=status,
        )

        worker_rank = status.source
        processed_images += int(result_meta[1])

        next_index, has_work = dispatch_work(
            comm,
            worker_rank,
            next_index,
            total_images,
            batch_size,
        )
        if not has_work:
            active_workers -= 1

    return processed_images


def worker(
    comm,
    image_names: list[str],
    input_dir: Path,
    output_dir: Path,
) -> None:
    while True:
        status = MPI.Status()
        recv_vec = np.empty(2, dtype=np.intc)
        comm.Recv([recv_vec, MPI.INT], source=ROOT, tag=MPI.ANY_TAG, status=status)

        if status.tag == STOP_TAG:
            break

        start_index = int(recv_vec[0])
        image_count = int(recv_vec[1])
        processed_count = process_batch(
            image_names,
            input_dir,
            output_dir,
            start_index,
            image_count,
        )
        result_meta = np.array([start_index, processed_count], dtype=np.intc)
        comm.Send([result_meta, MPI.INT], dest=ROOT, tag=RESULT_META_TAG)


def main() -> None:
    if not MPI.Is_initialized():
        MPI.Init()

    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == ROOT:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        image_names = list_input_images(args.input_dir)
        if not image_names:
            raise FileNotFoundError(
                f"Aucune image JPG trouvee dans {args.input_dir}"
            )
    else:
        image_names = None

    input_dir_str = comm.bcast(str(args.input_dir), root=ROOT)
    output_dir_str = comm.bcast(str(args.output_dir), root=ROOT)
    image_names = comm.bcast(image_names, root=ROOT)
    total_images = len(image_names)

    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    comm.Barrier()

    start_time = perf_counter()

    if size == 1:
        processed_images = process_batch(
            image_names,
            input_dir,
            output_dir,
            0,
            total_images,
        )
    elif rank == ROOT:
        processed_images = master(comm, size, total_images, args.batch_size)
    else:
        worker(comm, image_names, input_dir, output_dir)
        processed_images = 0

    elapsed_time = perf_counter() - start_time

    if rank == ROOT:
        if size == 1:
            processed_images = total_images
        print(f"Images traitees : {processed_images}/{total_images}")
        print(f"Temps total de traitement : {elapsed_time:.6f} s")


if __name__ == "__main__":
    main()
