#!/usr/bin/env python3
"""
Minimal RX+TX+gossip loop for holographic UDP chunks.

The node listens for datagrams, stores complete chunks, forwards a fraction
of newly completed chunks, and periodically re-sends a random chunk from its
local store. This keeps the field "alive" without streams or sessions.

Example:
  python3 examples/holo_mesh_node.py \\
      --bind 0.0.0.0:5000 \\
      --peer 10.0.0.2:5000 --peer 10.0.0.3:5000 \\
      --store-root cortex_node \\
      --rate-hz 40 --forward-prob 0.3 \\
      --seed-uri holo://demo/flower --seed-chunk-dir flower.jpg.holo
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holo.cortex.store import CortexStore
from holo.net.arch import content_id_bytes_from_uri
from holo.net.mesh import MeshNode
from holo.net.transport import iter_chunk_datagrams


def _parse_addr(addr: str) -> Tuple[str, int]:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _chunk_id_from_name(name: str) -> int:
    if name.startswith("chunk_") and name.endswith(".holo"):
        try:
            return int(name[6:-5])
        except ValueError:
            return 0
    return 0


def _send_chunk_bytes(
    sock: socket.socket,
    peers: Iterable[Tuple[str, int]],
    *,
    content_id: bytes,
    chunk_id: int,
    chunk_bytes: bytes,
    max_payload: int,
    auth_key: Optional[bytes],
) -> None:
    for peer in peers:
        for dg in iter_chunk_datagrams(
            content_id,
            chunk_id,
            chunk_bytes,
            max_payload=max_payload,
            auth_key=auth_key,
        ):
            sock.sendto(dg, peer)


def _seed_store(store: CortexStore, content_uri: str, chunk_dir: Path) -> None:
    """
    Optional bootstrap: inject an existing chunk directory into this node's store.
    """
    cid = content_id_bytes_from_uri(content_uri)
    for path in sorted(chunk_dir.glob("chunk_*.holo")):
        chunk_id = _chunk_id_from_name(path.name)
        if chunk_id < 0:
            continue
        store.store_chunk_bytes(cid, chunk_id, path.read_bytes())


def _radiate_random_chunk(
    store_root: str,
    sock: socket.socket,
    node: MeshNode,
) -> None:
    cdirs = [
        os.path.join(store_root, d)
        for d in os.listdir(store_root)
        if os.path.isdir(os.path.join(store_root, d))
    ]
    if not cdirs:
        return
    cdir = Path(random.choice(cdirs))
    chunks = list(cdir.glob("chunk_*.holo"))
    if not chunks:
        return
    path = random.choice(chunks)
    cid = bytes.fromhex(cdir.name)
    chunk_id = _chunk_id_from_name(path.name)
    chunk_bytes = path.read_bytes()
    _send_chunk_bytes(
        node.sock,
        node.peers,
        content_id=cid,
        chunk_id=chunk_id,
        chunk_bytes=chunk_bytes,
        max_payload=node.max_payload,
        auth_key=node.auth_key,
    )


def run_node(
    bind: Tuple[str, int],
    peers: List[Tuple[str, int]],
    store_root: str,
    *,
    rate_hz: float = 50.0,
    forward_prob: float = 0.25,
    auth_key: Optional[bytes] = None,
    max_payload: int = 1200,
) -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(bind)
        sock.setblocking(False)
    except PermissionError as exc:  # pragma: no cover
        raise SystemExit(f"socket bind failed (perm?): {exc}") from exc

    store = CortexStore(store_root)
    node = MeshNode(sock, store, peers=peers, auth_key=auth_key, max_payload=max_payload)

    seen = set()
    t_next = time.time()

    try:
        while True:
            # RX pump
            while True:
                res = node.recv_once()
                if not res:
                    break
                cid, chunk_id, path = res
                key = (cid, int(chunk_id))
                if key in seen:
                    continue
                seen.add(key)

                if random.random() < forward_prob and node.peers:
                    chunk_bytes = Path(path).read_bytes()
                    _send_chunk_bytes(
                        node.sock,
                        node.peers,
                        content_id=cid,
                        chunk_id=int(chunk_id),
                        chunk_bytes=chunk_bytes,
                        max_payload=node.max_payload,
                        auth_key=node.auth_key,
                    )

            # TX radiation loop
            now = time.time()
            if now >= t_next and node.peers:
                _radiate_random_chunk(store_root, sock, node)
                t_next = now + (1.0 / max(1e-6, rate_hz))

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Minimal holographic mesh node (RX+TX+gossip).")
    ap.add_argument("--bind", default="0.0.0.0:5000", help="ip:port to bind for UDP")
    ap.add_argument("--peer", action="append", default=[], help="peer ip:port (repeatable)")
    ap.add_argument("--store-root", default="cortex_node", help="where to persist chunks")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="retransmission rate from store")
    ap.add_argument("--forward-prob", type=float, default=0.25, help="probability to forward new chunks")
    ap.add_argument("--max-payload", type=int, default=1200, help="UDP payload budget")
    ap.add_argument("--auth-key", type=str, default=None, help="Optional hex key for HMAC")
    ap.add_argument(
        "--seed-uri",
        type=str,
        default=None,
        help="holo:// URI for seed content (used with --seed-chunk-dir)",
    )
    ap.add_argument(
        "--seed-chunk-dir",
        type=str,
        default=None,
        help="Existing .holo directory to inject into the store",
    )
    args = ap.parse_args(argv)

    bind = _parse_addr(args.bind)
    peers = [_parse_addr(p) for p in args.peer]
    auth_key = bytes.fromhex(args.auth_key) if args.auth_key else None

    if args.seed_chunk_dir and not args.seed_uri:
        ap.error("--seed-chunk-dir requires --seed-uri to map to a content_id")

    if args.seed_chunk_dir and args.seed_uri:
        store = CortexStore(args.store_root)
        _seed_store(store, args.seed_uri, Path(args.seed_chunk_dir))

    run_node(
        bind,
        peers,
        args.store_root,
        rate_hz=args.rate_hz,
        forward_prob=args.forward_prob,
        auth_key=auth_key,
        max_payload=args.max_payload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
