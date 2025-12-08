"""
Basic timing benchmark for encode/decode.

Measures wall-clock time for image and optional audio encode/decode paths.

Usage:
    python3 examples/timing_benchmark.py --image flower.jpg --audio examples/data/track.wav
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import holo  # noqa: E402


def time_one(label: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def bench_image(image: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="holographix_timing_img_") as td:
        chunk_dir = Path(td) / f"{image.name}.holo"
        recon = Path(td) / "recon.png"

        enc_t = time_one("encode_image", lambda: holo.encode_image_holo_dir(str(image), str(chunk_dir), target_chunk_kb=32))
        dec_t = time_one("decode_image", lambda: holo.decode_image_holo_dir(str(chunk_dir), str(recon)))

        print(f"[image] encode -> {enc_t*1e3:.1f} ms, decode -> {dec_t*1e3:.1f} ms")


def bench_audio(audio: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="holographix_timing_aud_") as td:
        chunk_dir = Path(td) / f"{audio.name}.holo"
        recon = Path(td) / "recon.wav"

        enc_t = time_one("encode_audio", lambda: holo.encode_audio_holo_dir(str(audio), str(chunk_dir), target_chunk_kb=32))
        dec_t = time_one("decode_audio", lambda: holo.decode_audio_holo_dir(str(chunk_dir), str(recon)))

        print(f"[audio] encode -> {enc_t*1e3:.1f} ms, decode -> {dec_t*1e3:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Timing benchmark for holographix encode/decode.")
    ap.add_argument("--image", type=str, help="Image path to test.")
    ap.add_argument("--audio", type=str, help="Audio WAV path to test.")
    args = ap.parse_args()

    if not args.image and not args.audio:
        ap.error("Provide --image and/or --audio")

    if args.image:
        path = Path(args.image)
        if not path.exists():
            raise FileNotFoundError(path)
        bench_image(path)

    if args.audio:
        path = Path(args.audio)
        if not path.exists():
            raise FileNotFoundError(path)
        bench_audio(path)


if __name__ == "__main__":
    main()
