#!/usr/bin/env python3
"""
Run fixed-point healing and print convergence.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "no-signal.jpg"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holo
from holo.field import Field
from PIL import Image


def _make_target(path: Path, size: int = 256) -> None:
    if path.exists():
        return
    img = Image.new("RGB", (size, size), (40, 90, 180))
    for y in range(size):
        for x in range(size):
            img.putpixel((x, y), ((40 + x) % 255, (90 + y) % 255, (180 + x + y) % 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    base = Path(__file__).parent
    out_dir = base / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAMPLE_IMAGE.exists():
        src = SAMPLE_IMAGE
    else:
        src = out_dir / "healing_target.png"
        _make_target(src)

    chunk_dir = out_dir / f"{Path(src).stem}_fp.holo"
    holo.encode_image_holo_dir(str(src), str(chunk_dir), target_chunk_kb=12)

    # Remove a few chunks to simulate loss
    chunk_files = list(chunk_dir.glob("chunk_*.holo"))
    drop = max(1, len(chunk_files) // 4) if chunk_files else 0
    for path in random.sample(chunk_files, k=drop):
        path.unlink(missing_ok=True)

    field = Field(content_id="demo/fixed_point", chunk_dir=str(chunk_dir))
    report = field.heal_fixed_point(str(out_dir / f"{Path(src).stem}_fp_healed.holo"), max_iters=4, tol=1e-3)

    print("=== heal_fixed_point ===")
    print(f"iterations: {report['iterations']}")
    print("deltas:", report["deltas"])
    print("drift :", report["drift"])

    # Write reconstructions for each iteration
    for idx, d in enumerate(report.get("dirs", []), start=1):
        recon_path = out_dir / f"{Path(src).stem}_fp_iter{idx}.png"
        try:
            holo.decode_image_holo_dir(d, str(recon_path))
            print(f"iter {idx}: {recon_path}")
        except Exception:
            continue


if __name__ == "__main__":
    main()
