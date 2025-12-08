import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from holo.codec import (
    _golden_permutation,
    decode_audio_holo_dir,
    decode_image_holo_dir,
    encode_audio_holo_dir,
    encode_image_holo_dir,
)
from holo.net.mesh import MeshNode
from holo.cortex.store import CortexStore


class TestGoldenPermutation(unittest.TestCase):
    def test_full_cycle(self) -> None:
        for n in [1, 2, 3, 10, 64, 101, 256]:
            perm = _golden_permutation(n)
            self.assertEqual(perm.shape, (n,))
            self.assertTrue(np.array_equal(np.sort(perm), np.arange(n)))


class TestRoundTrip(unittest.TestCase):
    def test_image_round_trip_exact(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(17, 11, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "img.png"
            recon = tmp / "img_recon.png"
            chunk_dir = tmp / "img.holo"

            Image.fromarray(img, mode="RGB").save(src)
            encode_image_holo_dir(str(src), str(chunk_dir), block_count=8)
            decode_image_holo_dir(str(chunk_dir), str(recon))

            out = np.asarray(Image.open(recon).convert("RGB"), dtype=np.uint8)
            np.testing.assert_array_equal(img, out)

    def test_audio_round_trip_exact(self) -> None:
        sr = 8000
        n_frames = 512
        freq = 440.0
        t = np.arange(n_frames, dtype=np.float64) / sr
        # Keep amplitude moderate to avoid clipping flags.
        tone = (np.sin(2.0 * math.pi * freq * t) * 12000).astype(np.int16)
        audio = tone.reshape(-1, 1)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "tone.wav"
            recon = tmp / "tone_recon.wav"
            chunk_dir = tmp / "tone.holo"

            _write_wav(src, audio, sr)
            encode_audio_holo_dir(str(src), str(chunk_dir), block_count=6, coarse_max_frames=64)
            decode_audio_holo_dir(str(chunk_dir), str(recon))

            out, out_sr, out_ch = _read_wav(recon)
            self.assertEqual(out_sr, sr)
            self.assertEqual(out_ch, 1)
            np.testing.assert_array_equal(audio, out)

    def test_image_degradation_monotone_psnr(self) -> None:
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "img.png"
            chunk_dir = tmp / "img.holo"

            Image.fromarray(img, mode="RGB").save(src)
            encode_image_holo_dir(str(src), str(chunk_dir), block_count=12)

            full = np.asarray(Image.fromarray(img, mode="RGB"), dtype=np.uint8)
            psnr_values = []
            for k in [1, 3, 6, 12]:
                recon = _decode_image_subset(chunk_dir, k)
                psnr_values.append(_psnr(full, recon))

        # PSNR should be non-decreasing as k grows
        for a, b in zip(psnr_values, psnr_values[1:]):
            self.assertGreaterEqual(b + 1e-6, a)

    def test_corrupt_chunks_still_decode_image(self) -> None:
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "img.png"
            chunk_dir = tmp / "img.holo"
            recon = tmp / "img_recon.png"

            Image.fromarray(img, mode="RGB").save(src)
            encode_image_holo_dir(str(src), str(chunk_dir), block_count=10, coarse_max_side=8)

            # delete a few chunks to simulate loss
            to_remove = list(sorted(chunk_dir.glob("chunk_*.holo")))[:3]
            for p in to_remove:
                p.unlink()

            decode_image_holo_dir(str(chunk_dir), str(recon))
            out = np.asarray(Image.open(recon).convert("RGB"), dtype=np.uint8)
            self.assertEqual(out.shape, img.shape)

            psnr_lossy = _psnr(img, out)
            # With missing chunks quality should be worse than exact
            self.assertLess(psnr_lossy, 100.0)

    def test_mesh_udp_loopback(self) -> None:
        rng = np.random.default_rng(3)
        img = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "img.png"
            chunk_dir = tmp / "img.holo"
            out_store = tmp / "cortex_rx"

            Image.fromarray(img, mode="RGB").save(src)
            encode_image_holo_dir(str(src), str(chunk_dir), block_count=4)

            # set up loopback sockets
            recv_sock = _udp_sock(("127.0.0.1", 0))
            send_sock = _udp_sock(("127.0.0.1", 0))
            recv_addr = recv_sock.getsockname()

            rx = MeshNode(recv_sock, CortexStore(str(out_store)))
            tx = MeshNode(send_sock, CortexStore(str(tmp / "cortex_tx")), peers=[recv_addr])

            tx.broadcast_chunk_dir("holo://test/image", str(chunk_dir))

            cdir = Path(out_store)
            for _ in range(200):
                res = rx.recv_once()
                if res is None:
                    continue
                cid, chunk_id, stored = res
                self.assertTrue(Path(stored).exists())
                cid_dir = cdir / cid.hex()
                if cid_dir.exists() and len(list(cid_dir.glob("chunk_*.holo"))) >= 4:
                    break

            stored_chunks = sorted(cdir.glob("**/chunk_*.holo"))
            self.assertGreaterEqual(len(stored_chunks), 1)

            recv_sock.close()
            send_sock.close()


def _udp_sock(bind_addr: tuple[str, int]) -> "socket.socket":
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(bind_addr)
    sock.setblocking(False)
    return sock


def _decode_image_subset(chunk_dir: Path, k: int) -> np.ndarray:
    subset = sorted(chunk_dir.glob("chunk_*.holo"))[: max(1, int(k))]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for src in subset:
            tmp_path = tmp / src.name
            tmp_path.write_bytes(src.read_bytes())
        recon_path = tmp / "recon.png"
        decode_image_holo_dir(str(tmp), str(recon_path))
        return np.asarray(Image.open(recon_path).convert("RGB"), dtype=np.uint8)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mse = np.mean((a_f - b_f) ** 2)
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.int16)
    n_frames, ch = samples.shape
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(ch))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(samples.astype("<i2", copy=False).tobytes())


def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, ch)
    return data.astype(np.int16, copy=False), int(sr), int(ch)


if __name__ == "__main__":
    unittest.main()
