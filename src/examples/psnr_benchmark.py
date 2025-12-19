"""
PSNR benchmark for holographic images under partial chunk coverage.

This script encodes an image into holographic chunks, then repeatedly
decodes random subsets of those chunks to measure how reconstruction
quality (PSNR) scales with the number of received fragments.

Usage (from repo root):
    python3 examples/psnr_benchmark.py --image flower.jpg --target-chunk-kb 32
"""

import argparse
import csv
import math
import random
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holo.codec import _decode_image_holo_core, encode_image_holo_dir, load_image_rgb_u8


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)
    mse = np.mean((ref_f - test_f) ** 2)
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _unique_k_values(total: int, requested: Sequence[int]) -> List[int]:
    keep = []
    seen = set()
    for k in requested:
        k = min(total, max(1, int(k)))
        if k not in seen:
            keep.append(k)
            seen.add(k)
    return keep


def default_k_values(total: int) -> List[int]:
    if total <= 0:
        return []
    base = [1, max(1, total // 4), max(1, total // 2), total]
    return _unique_k_values(total, base)


def run_trials(
    chunk_paths: Sequence[Path],
    ref_img: np.ndarray,
    k_values: Iterable[int],
    samples_per_k: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    results = []

    for k in k_values:
        k = min(len(chunk_paths), max(1, int(k)))
        psnr_values = []

        for _ in range(samples_per_k):
            subset = rng.sample(chunk_paths, k)
            with tempfile.TemporaryDirectory() as td:
                subset_dir = Path(td)
                for src in subset:
                    shutil.copyfile(src, subset_dir / src.name)
                recon = _decode_image_holo_core(str(subset_dir))
            psnr_values.append(psnr(ref_img, recon))

        results.append(_psnr_stats(k, psnr_values))

    return results


def _psnr_stats(k: int, values: Sequence[float]) -> dict:
    vals = np.array(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    has_inf = np.isinf(vals).any()

    if finite.size == 0:
        return {"k": k, "mean_psnr": float("inf"), "std_psnr": 0.0, "min_psnr": float("inf"), "max_psnr": float("inf")}

    mean = float("inf") if has_inf else float(finite.mean())
    std = 0.0 if has_inf else float(finite.std(ddof=0))
    min_v = float(finite.min())
    max_v = float("inf") if has_inf else float(finite.max())

    return {"k": k, "mean_psnr": mean, "std_psnr": std, "min_psnr": min_v, "max_psnr": max_v}


def print_results(results: List[dict]) -> None:
    if not results:
        print("No results to show.")
        return

    print("k\tmean_psnr(dB)\tstd(dB)\tmin(dB)\tmax(dB)")
    for row in results:
        print(
            f"{row['k']}\t"
            f"{row['mean_psnr']:.3f}\t"
            f"{row['std_psnr']:.3f}\t"
            f"{row['min_psnr']:.3f}\t"
            f"{row['max_psnr']:.3f}"
        )


def write_csv(path: Path, results: List[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "mean_psnr", "std_psnr", "min_psnr", "max_psnr"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="PSNR benchmark for holographic images.")
    parser.add_argument("--image", type=str, default="flower.jpg", help="Input image path.")
    parser.add_argument("--target-chunk-kb", type=int, default=32, help="Target chunk size in KB (optional).")
    parser.add_argument(
        "--block-count",
        type=int,
        default=None,
        help="Force a specific number of chunks (overrides target-chunk-kb if set).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=None,
        help="Explicit k values (number of chunks) to test. Defaults to [1, B/4, B/2, B].",
    )
    parser.add_argument("--samples-per-k", type=int, default=8, help="Number of random subsets per k.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for subset selection.")
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Directory to write the temporary .holo field (defaults to a temp directory).",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional path to write results as CSV.")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    tempdir_ctx = None
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tempdir_ctx = tempfile.TemporaryDirectory(prefix="holographix_psnr_")
        workdir = Path(tempdir_ctx.name)

    chunk_dir = workdir / f"{image_path.name}.holo"
    encode_image_holo_dir(
        str(image_path),
        str(chunk_dir),
        block_count=args.block_count if args.block_count is not None else 32,
        target_chunk_kb=args.target_chunk_kb if args.block_count is None else None,
    )

    chunk_paths = sorted(chunk_dir.glob("chunk_*.holo"))
    if not chunk_paths:
        raise RuntimeError(f"No chunks were written to {chunk_dir}")

    ref_img = load_image_rgb_u8(str(image_path))
    k_values = _unique_k_values(len(chunk_paths), args.k) if args.k else default_k_values(len(chunk_paths))
    results = run_trials(chunk_paths, ref_img, k_values, args.samples_per_k, args.seed)

    print(f"Encoded {len(chunk_paths)} chunks to {chunk_dir}")
    print_results(results)

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(csv_path, results)
        print(f"Wrote CSV to {csv_path}")

    if tempdir_ctx is not None:
        tempdir_ctx.cleanup()


if __name__ == "__main__":
    main()
