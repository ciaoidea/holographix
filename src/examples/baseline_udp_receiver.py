"""
Simple UDP baseline receiver: reassemble datagrams into a file.

Usage:
  python3 examples/baseline_udp_receiver.py --listen 0.0.0.0:6000 --out received.bin --duration 5
"""

import argparse
import socket
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline UDP file receiver (no FEC, no retransmit).")
    ap.add_argument("--listen", default="0.0.0.0:6000", help="ip:port to bind")
    ap.add_argument("--out", required=True, help="output path")
    ap.add_argument("--duration", type=float, default=5.0, help="seconds to listen")
    args = ap.parse_args()

    host, port = args.listen.split(":")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, int(port)))
    sock.setblocking(False)

    frags = {}
    total = None
    t0 = time.time()
    while time.time() - t0 < args.duration:
        try:
            data, _ = sock.recvfrom(65536)
        except BlockingIOError:
            time.sleep(0.01)
            continue
        if len(data) < 8:
            continue
        idx = int.from_bytes(data[:4], "big")
        tot = int.from_bytes(data[4:8], "big")
        total = tot if total is None else total
        frags[idx] = data[8:]
    if total is None:
        print("no datagrams received")
        return
    ordered = [frags.get(i, b"") for i in range(total)]
    payload = b"".join(ordered)
    Path(args.out).write_bytes(payload)
    received = sum(len(frags[k]) for k in frags)
    print(f"received {len(frags)}/{total} fragments; wrote {len(payload)} bytes (raw received {received} bytes)")


if __name__ == "__main__":
    main()
