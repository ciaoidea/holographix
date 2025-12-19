"""
Receive holographic chunks over UDP and decode best-effort.

Usage:
  python3 examples/holo_mesh_receiver.py --listen 0.0.0.0:5000 --out-dir rx_store --decode image_recon.png
"""

import argparse
import socket
import sys
import time
from pathlib import Path

from holo.cortex.store import CortexStore
from holo.field import Field
from holo.net.mesh import MeshNode


def parse_addr(addr: str) -> tuple[str, int]:
    host, port = addr.split(":")
    return host, int(port)


def main() -> None:
    ap = argparse.ArgumentParser(description="Receive holographic chunks over UDP and decode best view.")
    ap.add_argument("--listen", default="0.0.0.0:5000", help="ip:port to bind")
    ap.add_argument("--out-dir", default="cortex_rx", help="where to store received chunks")
    ap.add_argument("--decode", default=None, help="Path to write best decode (image/audio) on exit")
    ap.add_argument("--duration", type=float, default=5.0, help="how long to listen (seconds)")
    ap.add_argument("--auth-key", type=str, default=None, help="Optional hex key for HMAC (32 bytes hex)")
    args = ap.parse_args()

    bind_host, bind_port = parse_addr(args.listen)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, bind_port))
    sock.setblocking(False)

    key = bytes.fromhex(args.auth_key) if args.auth_key else None
    store = CortexStore(args.out_dir)
    mesh = MeshNode(sock, store, auth_key=key)

    t0 = time.time()
    while time.time() - t0 < args.duration:
        res = mesh.recv_once()
        if res:
            cid, chunk_id, path = res
            print(f"stored chunk {chunk_id} for cid {cid.hex()} at {path}")
        time.sleep(0.02)

    print(f"datagrams seen={mesh._assembler.counters['datagrams']} mac_fail={mesh._assembler.counters['mac_fail']} stored_chunks={mesh.counters['stored_chunks']}")

    if args.decode:
        # Decode best view of the first content_id observed
        content_dirs = list(Path(args.out_dir).glob("*"))
        if not content_dirs:
            sys.exit("no content received, nothing to decode")

        cid_hex = content_dirs[0].name
        cid_dir = content_dirs[0]
        mode = None
        try:
            # Peek magic byte to detect mode
            from holo.codec import detect_mode_from_chunk_dir

            mode = detect_mode_from_chunk_dir(str(cid_dir))
        except Exception:
            pass

        fid = f"holo://{cid_hex}"
        field = Field(content_id=fid, chunk_dir=str(cid_dir))
        if mode == "audio":
            field.best_decode_audio(args.decode)
        else:
            field.best_decode_image(args.decode)
        print(f"decoded best view to {args.decode}")


if __name__ == "__main__":
    main()
