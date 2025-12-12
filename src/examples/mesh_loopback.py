#!/usr/bin/env python3
"""
Minimal mesh demo: send holographic chunks over UDP to another node that
stores and decodes them by content identifier (holo://...).
"""

from __future__ import annotations

import socket
import time
import sys
from pathlib import Path

SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "galaxy.jpg"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import holo
from holo.net.arch import content_id_bytes_from_uri
from holo.net.mesh import MeshNode
from holo.cortex.store import CortexStore
from PIL import Image


def _make_spiral(path: Path, size: int = 256) -> None:
    if path.exists():
        return
    img = Image.new("RGB", (size, size), (10, 10, 40))
    cx = cy = size / 2.0
    for i in range(size * 4):
        t = i / float(size * 4)
        angle = 12.0 * t
        r = (size / 2.0) * t
        x = int(cx + r * (2 ** 0.5) * t * 0.4 + r * 0.5 * t)
        y = int(cy + r * 0.5 * t)
        if 0 <= x < size and 0 <= y < size:
            img.putpixel((x, y), (220, 120 + int(80 * t), 60 + int(180 * t)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"
    out_dir = base / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if SAMPLE_IMAGE.exists():
        img_path = SAMPLE_IMAGE
    else:
        img_path = data_dir / "spiral.png"
        _make_spiral(img_path)

    chunk_dir = out_dir / f"{Path(img_path).stem}.holo"
    holo.encode_image_holo_dir(str(img_path), str(chunk_dir), target_chunk_kb=16)

    # Two nodes on localhost
    sock_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_a.bind(("127.0.0.1", 0))
    sock_b.bind(("127.0.0.1", 0))
    sock_a.setblocking(False)
    sock_b.setblocking(False)

    addr_a = sock_a.getsockname()
    addr_b = sock_b.getsockname()

    store_a = CortexStore(str(out_dir / "store_a"))
    store_b = CortexStore(str(out_dir / "store_b"))
    mesh_a = MeshNode(sock_a, store_a, peers=[addr_b])
    mesh_b = MeshNode(sock_b, store_b, peers=[addr_a])

    content_uri = f"holo://demo/{Path(img_path).stem}"
    content_id = content_id_bytes_from_uri(content_uri)

    # Send all chunks from A to B
    mesh_a.broadcast_chunk_dir(content_uri, str(chunk_dir))

    # Receive on B for a short window
    t0 = time.time()
    while time.time() - t0 < 2.0:
        mesh_b.recv_once()
        time.sleep(0.02)

    received_dir = Path(store_b.content_dir(content_id))
    recon = out_dir / f"{Path(img_path).stem}_mesh_recon.png"
    holo.decode_image_holo_dir(str(received_dir), str(recon))

    print("=== mesh_loopback ===")
    print(f"sender chunks : {chunk_dir} -> broadcast to {addr_b}")
    print(f"receiver store: {received_dir}")
    print(f"reconstructed : {recon}")
    print(f"Chunks are addressed by {content_uri} content ID, not socket endpoint.")


if __name__ == "__main__":
    main()
