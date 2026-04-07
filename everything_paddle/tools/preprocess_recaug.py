"""Pre-apply PaddleOCR's RecAug to a batch of images in parallel.

Sends chunks of images to worker processes to minimize IPC overhead.

Usage:
    uv run python tools/preprocess_recaug.py --input DIR --output DIR [--variants N] [--workers N]
    uv run python tools/preprocess_recaug.py --benchmark --input DIR
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import cv2

_PROJECT_ROOT = str(Path(__file__).parent.parent)
_worker_aug = None


def _worker_init():
    global _worker_aug
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from paddleocr_repo.ppocr.data.imaug.rec_img_aug import RecAug
    _worker_aug = RecAug()


def augment_chunk(chunk):
    """Process a chunk of (input_path, label, output_dir, variants, base_idx) tuples."""
    global _worker_aug
    if _worker_aug is None:
        _worker_init()

    results = []
    for input_path, label, output_dir, variants, base_idx in chunk:
        img = cv2.imread(input_path)
        if img is None:
            continue
        for v in range(variants):
            data = {"image": img.copy()}
            out_img = _worker_aug(data)["image"]
            fname = f"aug_{base_idx:06d}_{v:02d}.png"
            cv2.imwrite(os.path.join(output_dir, fname), out_img)
            results.append((fname, label))
    return results


def make_chunks(tasks, n_chunks):
    """Split tasks into roughly equal chunks."""
    chunk_size = max(1, len(tasks) // n_chunks)
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]


def load_tasks(input_dir, max_samples=None):
    input_dir = Path(input_dir)
    train_list = input_dir / "train_list.txt"
    if not train_list.exists():
        print(f"No train_list.txt in {input_dir}")
        return []
    lines = train_list.read_text(encoding="utf-8").strip().split("\n")
    if max_samples:
        lines = lines[:max_samples]
    tasks = []
    for idx, line in enumerate(lines):
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        fname, label = parts
        tasks.append((str(input_dir / fname), label, idx))
    return tasks


def run_augmentation(input_dir, output_dir, variants, workers):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_tasks = load_tasks(input_dir)
    if not raw_tasks:
        return

    tasks = [(path, label, str(output_dir), variants, idx)
             for path, label, idx in raw_tasks]
    chunks = make_chunks(tasks, workers * 4)

    total_output = len(tasks) * variants
    print(f"Augmenting {len(tasks)} images x {variants} variants = {total_output} output images")
    print(f"Workers: {workers}, Chunks: {len(chunks)}")

    all_results = []
    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = {executor.submit(augment_chunk, chunk): len(chunk) for chunk in chunks}
        done_imgs = 0
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            done_imgs += futures[future]
            if done_imgs % 2000 < futures[future]:
                elapsed = time.perf_counter() - start
                rate = len(all_results) / elapsed
                print(f"  {done_imgs}/{len(tasks)} src imgs, "
                      f"{len(all_results)} aug imgs ({rate:.0f} aug/s)")

    elapsed = time.perf_counter() - start
    print(f"\nDone: {len(all_results)} augmented images in {elapsed:.1f}s "
          f"({len(all_results) / elapsed:.0f} imgs/s)")

    aug_lines = [f"{fname}\t{label}" for fname, label in all_results]
    (output_dir / "train_list.txt").write_text("\n".join(aug_lines), encoding="utf-8")

    input_dir = Path(input_dir)
    for f in ["dict.txt", "val_list.txt"]:
        src = input_dir / f
        if src.exists():
            (output_dir / f).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return len(all_results), elapsed


def run_benchmark(input_dir, max_samples=500):
    import shutil

    raw_tasks = load_tasks(input_dir, max_samples=max_samples)
    if not raw_tasks:
        return

    tmp_dir = Path("debug_ocr/_bench_tmp")
    max_cores = cpu_count()
    worker_counts = sorted(set(
        w for w in [1, 2, 4, 8, 12, 16, 24, 32, max_cores] if w <= max_cores
    ))

    print(f"Benchmarking RecAug (chunked) on {len(raw_tasks)} images, up to {max_cores} cores")
    print(f"{'Workers':>8} | {'Time (s)':>10} | {'imgs/s':>10} | {'Speedup':>8}")
    print(f"{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")

    baseline_rate = None

    for nw in worker_counts:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        tasks = [(path, label, str(tmp_dir), 1, idx)
                 for path, label, idx in raw_tasks]
        chunks = make_chunks(tasks, nw * 4)

        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw, initializer=_worker_init) as executor:
            list(executor.map(augment_chunk, chunks))
        elapsed = time.perf_counter() - start
        rate = len(tasks) / elapsed

        if baseline_rate is None:
            baseline_rate = rate
        speedup = rate / baseline_rate

        print(f"{nw:>8} | {elapsed:>10.2f} | {rate:>10.1f} | {speedup:>7.1f}x")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Pre-apply RecAug augmentation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench-samples", type=int, default=500)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.input, max_samples=args.bench_samples)
    else:
        if args.output is None:
            print("--output is required")
            return
        workers = args.workers or cpu_count()
        run_augmentation(args.input, args.output, args.variants, workers)


if __name__ == "__main__":
    main()
