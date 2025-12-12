"""
Simple UDP baseline sender: send a file in datagrams with a small header.

Usage:
  python3 examples/baseline_udp_sender.py --file flower.jpg --peer 10.10.3.2:6000 --payload 1200
"""

import argparse
import socket
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline UDP file sender (no holography, no retransmit).")
    ap.add_argument("--file", required=True, help="Path to file to send.")
    ap.add_argument("--peer", required=True, help="peer ip:port")
    ap.add_argument("--payload", type=int, default=1200, help="max datagram size")
    args = ap.parse_args()

    peer_host, peer_port = args.peer.split(":")
    data = Path(args.file).read_bytes()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    payload_size = max(1, args.payload - 8)
    total = (len(data) + payload_size - 1) // payload_size
    sent = 0
    for idx in range(total):
        chunk = data[idx * payload_size : (idx + 1) * payload_size]
        header = idx.to_bytes(4, "big") + total.to_bytes(4, "big")
        sock.sendto(header + chunk, (peer_host, int(peer_port)))
        sent += 1
    print(f"sent {sent} datagrams for {len(data)} bytes")


if __name__ == "__main__":
    main()
