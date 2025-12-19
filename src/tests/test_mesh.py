from __future__ import annotations

import socket
import tempfile
import time
from pathlib import Path

from holo.cortex.store import CortexStore
from holo.net.arch import content_id_bytes_from_uri
from holo.net.mesh import MeshNode


def _udp_sock() -> socket.socket:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except PermissionError as exc:
        import unittest

        raise unittest.SkipTest(f"UDP socket not permitted: {exc}") from exc
    s.bind(("127.0.0.1", 0))
    s.setblocking(False)
    return s


def test_inv_want_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store_a = CortexStore(str(tmp / "cortex_a"))
        store_b = CortexStore(str(tmp / "cortex_b"))

        sock_a = _udp_sock()
        sock_b = _udp_sock()
        addr_a = sock_a.getsockname()
        addr_b = sock_b.getsockname()

        node_a = MeshNode(sock_a, store_a, peers=[addr_b], max_payload=1200)
        node_b = MeshNode(sock_b, store_b, peers=[addr_a], max_payload=1200)

        cid = content_id_bytes_from_uri("holo://test/inv-want")

        payloads = {0: b"alpha", 1: b"beta", 2: b"gamma"}
        for chunk_id, data in payloads.items():
            store_a.store_chunk_bytes(cid, int(chunk_id), data)

        node_a.send_inventory(cid, peers=[addr_b])

        stored: set[int] = set()
        deadline = time.time() + 2.0

        while time.time() < deadline and stored != set(payloads.keys()):
            res_b = node_b.recv_once()
            if res_b is not None:
                cid_b, chunk_id_b, path_b = res_b
                assert cid_b == cid
                assert Path(path_b).exists()
                stored.add(int(chunk_id_b))

            node_a.recv_once()
            time.sleep(0.001)

        try:
            assert stored == set(payloads.keys())
        finally:
            sock_a.close()
            sock_b.close()

        cdir_b = Path(store_b.content_dir(cid))
        assert cdir_b.exists()
        assert len(list(cdir_b.glob("chunk_*.holo"))) >= 3
