"""
SNR benchmark for holographic audio under partial chunk coverage.

Encodes a WAV file, samples random chunk subsets, and reports SNR stats.

Usage:
    python3 examples/snr_benchmark_audio.py --wav examples/data/track.wav --block-count 12
"""

import argparse
import csv
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from holo.codec import decode_audio_holo_dir, encode_audio_holo_dir  # noqa: E402


def snr_db(ref: np.ndarray, test: np.ndarray) -> float:
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)
    noise = ref_f - test_f
    p_signal = np.mean(ref_f ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0.0:
        return float("inf")
    return 10.0 * math.log10(p_signal / p_noise)


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
    wav_path: Path,
    k_values: Iterable[int],
    samples_per_k: int,
    seed: int,
) -> List[dict]:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        ch = wf.getnchannels()
        n_frames = wf.getnframes()
        ref = np.frombuffer(wf.readframes(n_frames), dtype="<i2").reshape(-1, ch)

    rng = random.Random(seed)
    results = []

    for k in k_values:
        k = min(len(chunk_paths), max(1, int(k)))
        snr_values = []

        for _ in range(samples_per_k):
            subset = rng.sample(chunk_paths, k)
            with tempfile.TemporaryDirectory() as td:
                subset_dir = Path(td)
                for src in subset:
                    shutil.copyfile(src, subset_dir / src.name)
                recon_path = subset_dir / "out.wav"
                decode_audio_holo_dir(str(subset_dir), str(recon_path))
                with wave.open(str(recon_path), "rb") as wf:
                    ch_r = wf.getnchannels()
                    n_frames_r = wf.getnframes()
                    if ch_r != ch or n_frames_r != ref.shape[0]:
                        continue
                    rec = np.frombuffer(wf.readframes(n_frames_r), dtype="<i2").reshape(-1, ch)
                snr_values.append(snr_db(ref, rec))

        results.append(_snr_stats(k, snr_values))

    return results


def _snr_stats(k: int, values: Sequence[float]) -> dict:
    vals = np.array(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    has_inf = np.isinf(vals).any()

    if finite.size == 0:
        return {"k": k, "mean_snr": float("inf"), "std_snr": 0.0, "min_snr": float("inf"), "max_snr": float("inf")}

    mean = float("inf") if has_inf else float(finite.mean())
    std = 0.0 if has_inf else float(finite.std(ddof=0))
    min_v = float(finite.min())
    max_v = float("inf") if has_inf else float(finite.max())

    return {"k": k, "mean_snr": mean, "std_snr": std, "min_snr": min_v, "max_snr": max_v}


def print_results(results: List[dict]) -> None:
    if not results:
        print("No results to show.")
        return

    print("k\tmean_snr(dB)\tstd(dB)\tmin(dB)\tmax(dB)")
    for row in results:
        print(
            f"{row['k']}\t"
            f"{row['mean_snr']:.3f}\t"
            f"{row['std_snr']:.3f}\t"
            f"{row['min_snr']:.3f}\t"
            f"{row['max_snr']:.3f}"
        )


def write_csv(path: Path, results: List[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "mean_snr", "std_snr", "min_snr", "max_snr"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="SNR benchmark for holographic audio.")
    parser.add_argument("--wav", type=str, required=True, help="Input WAV path.")
    parser.add_argument("--target-chunk-kb", type=int, default=32, help="Target chunk size in KB (optional).")
    parser.add_argument("--block-count", type=int, default=None, help="Force chunk count (overrides target if set).")
    parser.add_argument("--k", type=int, nargs="*", default=None, help="Explicit k values to test.")
    parser.add_argument("--samples-per-k", type=int, default=4, help="Random subsets per k.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")

    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    with tempfile.TemporaryDirectory(prefix="holographix_snr_") as td:
        chunk_dir = Path(td) / f"{wav_path.name}.holo"
        encode_audio_holo_dir(
            str(wav_path),
            str(chunk_dir),
            block_count=args.block_count if args.block_count is not None else 16,
            target_chunk_kb=args.target_chunk_kb if args.block_count is None else None,
        )

        chunk_paths = sorted(chunk_dir.glob("chunk_*.holo"))
        if not chunk_paths:
            raise RuntimeError("No chunks written")

        k_values = _unique_k_values(len(chunk_paths), args.k) if args.k else default_k_values(len(chunk_paths))
        results = run_trials(chunk_paths, wav_path, k_values, args.samples_per_k, args.seed)

    print_results(results)

    if args.csv:
        write_csv(Path(args.csv), results)
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
