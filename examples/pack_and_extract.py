#!/usr/bin/env python3
"""
Pack multiple images into one holographic field, delete some chunks,
and extract one object to see how graceful degradation behaves.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holo
from PIL import Image, ImageDraw


def _make_checker(path: Path, size: int = 256, cells: int = 8, fg=(20, 150, 240), bg=(250, 230, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    cell = size // cells
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for y in range(cells):
        for x in range(cells):
            if (x + y) % 2 == 0:
                draw.rectangle((x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1), fill=fg)
    img.save(path)


def _make_stripes(path: Path, size: int = 256, bands: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    band_h = size / float(bands)
    for i in range(bands):
        top = int(round(i * band_h))
        bottom = int(round((i + 1) * band_h))
        color = (int(20 + 20 * i), int(40 + 15 * i), int(200 - 10 * i))
        draw.rectangle((0, top, size, bottom), fill=color)
    img.save(path)


def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"
    out_dir = base / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_a = data_dir / "checker.png"
    img_b = data_dir / "stripes.png"
    _make_checker(img_a)
    _make_stripes(img_b)

    packed_dir = out_dir / "scene.holo"
    holo.pack_objects_holo_dir([str(img_a), str(img_b)], str(packed_dir), target_chunk_kb=16)

    # Drop some chunks to simulate damage
    chunk_files = list(packed_dir.glob("chunk_*.holo"))
    drop = max(1, len(chunk_files) // 3) if chunk_files else 0
    for path in random.sample(chunk_files, k=drop):
        path.unlink(missing_ok=True)

    # Extract the second object (index 1) from the degraded field
    recon_path = out_dir / "stripes_recon.png"
    holo.unpack_object_from_holo_dir(str(packed_dir), 1, str(recon_path))

    print("=== pack_and_extract ===")
    print(f"packed field dir   : {packed_dir} (kept {len(chunk_files) - drop}/{len(chunk_files)} chunks)")
    print(f"extracted object 1 : {recon_path}")
    print("Both objects share one residual trajectory; losing chunks blurs both instead of killing one.")


if __name__ == "__main__":
    main()
