#!/usr/bin/env python3
"""
Encode an image into holographic chunks, delete some of them,
and decode again to show graceful degradation.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "flower.jpg"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holo
from PIL import Image


def _make_gradient(path: Path, size: int = 256) -> None:
    """Generate a simple RGB gradient so the example is self-contained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    img = Image.new("RGB", (size, size))
    pixels = []
    for y in range(size):
        for x in range(size):
            r = int(255 * x / float(size - 1))
            g = int(255 * y / float(size - 1))
            b = int(255 * (x + y) / float(2 * (size - 1)))
            pixels.append((r, g, b))
    img.putdata(pixels)
    img.save(path)


def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"
    out_dir = base / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAMPLE_IMAGE.exists():
        src = SAMPLE_IMAGE
    else:
        src = data_dir / "gradient.png"
        _make_gradient(src)

    stem = src.stem
    chunk_dir = out_dir / f"{stem}.holo"
    # Fresh encode
    holo.encode_image_holo_dir(str(src), str(chunk_dir), target_chunk_kb=16)

    # Simulate loss: drop ~40% of chunks at random
    chunk_files = list(chunk_dir.glob("chunk_*.holo"))
    drop = int(len(chunk_files) * 0.4)
    to_delete = random.sample(chunk_files, k=drop) if drop > 0 else []
    for path in to_delete:
        path.unlink(missing_ok=True)

    # Decode with the surviving chunks
    recon_path = out_dir / f"{stem}_recon.png"
    holo.decode_image_holo_dir(str(chunk_dir), str(recon_path))

    print("=== encode_and_corrupt ===")
    print(f"source image       : {src}")
    print(f"holo chunks dir    : {chunk_dir} (kept {len(chunk_files) - drop}/{len(chunk_files)})")
    print(f"reconstructed image: {recon_path}")
    print("Open the recon to see graceful loss (detail fades, structure remains).")


if __name__ == "__main__":
    main()
