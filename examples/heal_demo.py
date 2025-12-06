#!/usr/bin/env python3
"""
Demonstrate healing: re-encode the best current percept after chunk loss.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holo
from holo.field import Field
from PIL import Image


def _make_target(path: Path, size: int = 256) -> None:
    if path.exists():
        return
    img = Image.new("RGB", (size, size), (60, 80, 200))
    for y in range(size):
        for x in range(size):
            img.putpixel((x, y), (60, (80 + x) % 255, (200 + y) % 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"
    out_dir = base / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = data_dir / "healing_target.png"
    _make_target(src)

    chunk_dir = out_dir / "healing.holo"
    holo.encode_image_holo_dir(str(src), str(chunk_dir), target_chunk_kb=12)

    # Damage: remove a few chunks
    chunk_files = list(chunk_dir.glob("chunk_*.holo"))
    drop = max(1, len(chunk_files) // 4) if chunk_files else 0
    for path in random.sample(chunk_files, k=drop):
        path.unlink(missing_ok=True)

    degraded = out_dir / "healing_degraded.png"
    holo.decode_image_holo_dir(str(chunk_dir), str(degraded))

    field = Field(content_id="demo/heal", chunk_dir=str(chunk_dir))
    healed_dir = out_dir / "healing_healed.holo"
    field.heal_to(str(healed_dir), target_chunk_kb=12)
    healed_img = out_dir / "healing_healed.png"
    holo.decode_image_holo_dir(str(healed_dir), str(healed_img))

    print("=== heal_demo ===")
    print(f"original image   : {src}")
    print(f"damaged chunks   : kept {len(chunk_files) - drop}/{len(chunk_files)} -> {degraded}")
    print(f"healed holo dir  : {healed_dir}")
    print(f"healed image     : {healed_img}")
    print("Healing re-encodes the best percept to repopulate chunks and slow decay.")


if __name__ == "__main__":
    main()
