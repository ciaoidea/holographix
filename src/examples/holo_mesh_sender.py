"""
Send holographic chunks over UDP using MeshNode.

Usage:
  python3 examples/holo_mesh_sender.py --uri holo://demo/flower --chunk-dir flower.jpg.holo --peer 10.10.3.2:5000
"""

import argparse
import socket
import sys
from pathlib import Path

from holo.net.mesh import MeshNode
from holo.cortex.store import CortexStore


def parse_addr(addr: str) -> tuple[str, int]:
    host, port = addr.split(":")
    return host, int(port)


def main() -> None:
    ap = argparse.ArgumentParser(description="Send holographic chunks to a peer via MeshNode.")
    ap.add_argument("--uri", required=True, help="holo:// content URI")
    ap.add_argument("--chunk-dir", required=True, help="Path to .holo directory to send")
    ap.add_argument("--peer", required=True, help="peer ip:port")
    ap.add_argument("--max-payload", type=int, default=1200, help="UDP payload size budget")
    ap.add_argument("--repeats", type=int, default=1, help="Resend each chunk this many times")
    ap.add_argument("--priority", choices=["gain"], default=None, help="Optional chunk priority ordering")
    ap.add_argument("--send-recovery", action="store_true", help="Send recovery_*.holo chunks too")
    ap.add_argument("--auth-key", type=str, default=None, help="Optional hex key for HMAC (32 bytes hex)")
    args = ap.parse_args()

    chunk_dir = Path(args.chunk_dir)
    if not chunk_dir.exists():
        sys.exit(f"chunk dir not found: {chunk_dir}")

    key = bytes.fromhex(args.auth_key) if args.auth_key else None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mesh = MeshNode(sock, CortexStore("cortex_tx"), peers=[parse_addr(args.peer)], max_payload=args.max_payload, auth_key=key)

    mesh.broadcast_chunk_dir(
        args.uri,
        str(chunk_dir),
        repeats=args.repeats,
        priority=args.priority,
        send_recovery=args.send_recovery,
    )
    print(f"sent {mesh.counters['sent_chunks']} chunks as datagrams={mesh.counters['sent_datagrams']}")


if __name__ == "__main__":
    main()
