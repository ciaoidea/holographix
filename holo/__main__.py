"""
holo.__main__

CLI entry point.

This file is intentionally small:
- parse args
- dispatch to the stable public API in holo.__init__
It must not contain codec math, storage policy, or networking logic.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import holo


def _default_decode_output(in_dir: str, mode: str) -> str:
    base = in_dir[:-5] if in_dir.lower().endswith(".holo") else in_dir
    root, ext = os.path.splitext(base)

    if mode == "audio":
        return root + "_recon.wav"

    if not ext:
        ext = ".png"
    return root + "_recon" + ext


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo",
        description="Holographix holographic codec: encode to .holo dirs, decode from .holo dirs.",
    )

    p.add_argument("path", help="Input file (image or .wav) to encode, or .holo directory to decode.")
    p.add_argument("target_kb", nargs="?", type=int, help="Target chunk size in KB (encoding only). Example: 32")
    p.add_argument("-o", "--output", default=None, help="Output path for decoding (default: derived from dir name).")
    p.add_argument("--max-chunks", type=int, default=None, help="Decode using at most this many chunks (debug).")

    p.add_argument("--blocks", type=int, default=None, help="Explicit chunk count for encoding.")
    p.add_argument("--coarse-side", type=int, default=64, help="Image coarse max side (encoding).")
    p.add_argument("--coarse-frames", type=int, default=2048, help="Audio coarse max frames (encoding).")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    path = os.path.normpath(args.path)

    if os.path.isdir(path) and path.lower().endswith(".holo"):
        mode = holo.detect_mode_from_chunk_dir(path)
        out = args.output or _default_decode_output(path, mode)

        if mode == "audio":
            holo.decode_audio_holo_dir(path, out, max_chunks=args.max_chunks)
        else:
            holo.decode_image_holo_dir(path, out, max_chunks=args.max_chunks)

        print(out)
        return 0

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mode = holo.detect_mode_from_extension(path)
    out_dir = path + ".holo"

    encode_kwargs = {}
    if args.target_kb is not None:
        encode_kwargs["target_chunk_kb"] = int(args.target_kb)
    if args.blocks is not None:
        encode_kwargs["block_count"] = int(args.blocks)

    if mode == "audio":
        encode_kwargs.setdefault("coarse_max_frames", int(args.coarse_frames))
        holo.encode_audio_holo_dir(path, out_dir, **encode_kwargs)
    else:
        encode_kwargs.setdefault("coarse_max_side", int(args.coarse_side))
        holo.encode_image_holo_dir(path, out_dir, **encode_kwargs)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
