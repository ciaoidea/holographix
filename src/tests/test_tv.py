import json
import tempfile
import unittest
from pathlib import Path

from holo.cortex.store import CortexStore
from holo.net.arch import content_id_bytes_from_stream_frame
from holo.net.transport import iter_chunk_datagrams
from holo.tv import HoloTVWindow, HoloTVFrame, HoloTVReceiver


class TestHoloTVWindow(unittest.TestCase):
    def test_round_robin_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            frame0 = tmp / "f0.holo"
            frame1 = tmp / "f1.holo"
            frame0.mkdir()
            frame1.mkdir()

            for idx in range(3):
                (frame0 / f"chunk_{idx:04d}.holo").write_bytes(b"f0" + bytes([idx]))
                (frame1 / f"chunk_{idx:04d}.holo").write_bytes(b"f1" + bytes([idx]))

            manifest = {
                "kind": "chunk_manifest",
                "manifest_version": 1,
                "base_kind": "image",
                "codec_version": 3,
                "block_count": 3,
                "ordered_chunks": [
                    "chunk_0002.holo",
                    "chunk_0000.holo",
                    "chunk_0001.holo",
                ],
                "chunks": [],
            }
            with open(frame0 / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            scores = {0: 1.0, 1: 3.0, 2: 2.0}
            for idx, score in scores.items():
                meta_path = frame1 / f"chunk_{idx:04d}.holo.meta"
                meta_path.write_text(f"{score:.6f}", encoding="ascii")

            window = HoloTVWindow(
                "holo://tv/test",
                [HoloTVFrame(0, str(frame0)), HoloTVFrame(1, str(frame1))],
                prefer_gain=True,
            )

            order = [(ref.frame_idx, Path(ref.path).name) for ref in window.iter_chunk_refs(order="round_robin")]
            expected = [
                (0, "chunk_0002.holo"),
                (1, "chunk_0001.holo"),
                (0, "chunk_0000.holo"),
                (1, "chunk_0002.holo"),
                (0, "chunk_0001.holo"),
                (1, "chunk_0000.holo"),
            ]
            self.assertEqual(order, expected)


class TestHoloTVReceiver(unittest.TestCase):
    def test_datagram_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = CortexStore(td)
            stream_id = "holo://tv/stream"
            frame_idx = 7
            chunk_id = 5
            payload = b"hello holotv"

            rx = HoloTVReceiver(stream_id, store, frame_indices=[frame_idx])
            content_id = content_id_bytes_from_stream_frame(stream_id, frame_idx)

            receipt = None
            for datagram in iter_chunk_datagrams(content_id, chunk_id, payload, max_payload=64):
                res = rx.push_datagram(datagram)
                if res is not None:
                    receipt = res

            self.assertIsNotNone(receipt)
            assert receipt is not None
            self.assertEqual(receipt.frame_idx, frame_idx)
            stored = Path(receipt.stored_path)
            self.assertTrue(stored.is_file())
            self.assertEqual(stored.read_bytes(), payload)
