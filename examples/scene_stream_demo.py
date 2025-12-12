#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holo.codec import decode_audio_holo_dir, decode_image_holo_dir, encode_audio_holo_dir, encode_image_holo_dir
from holo.cortex.store import CortexStore
from holo.net.arch import content_id_bytes_from_stream_frame
from holo.net.mesh import MeshNode
from holo.net.transport import iter_chunk_datagrams


def _parse_addr(s: str) -> Tuple[str, int]:
    host, port = s.rsplit(":", 1)
    return host, int(port)


def _chunk_id_from_name(name: str) -> int:
    if name.startswith("chunk_") and name.endswith(".holo"):
        try:
            return int(name[6:-5])
        except ValueError:
            return 0
    return 0


def _make_fallback_image(path: Path, *, size: int = 320) -> None:
    from PIL import Image

    x = np.linspace(0.0, 1.0, size, dtype=np.float64)[None, :]
    y = np.linspace(0.0, 1.0, size, dtype=np.float64)[:, None]
    r = (255.0 * x).astype(np.uint8)
    g = (255.0 * y).astype(np.uint8)
    b = (255.0 * (0.5 + 0.5 * np.sin(2.0 * math.pi * (x + y)))).astype(np.uint8)
    img = np.stack([r.repeat(size, axis=0), g.repeat(size, axis=1), b], axis=2)
    Image.fromarray(img, mode="RGB").save(path)


def _write_tone_wav(path: Path, *, seconds: float = 1.0, sr: int = 22050, freq_hz: float = 440.0) -> None:
    import wave

    n = max(2, int(round(seconds * sr)))
    t = np.arange(n, dtype=np.float64) / float(sr)
    s = 0.2 * np.sin(2.0 * math.pi * float(freq_hz) * t)
    audio = np.clip(np.round(s * 32767.0), -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio.tobytes())


def _send_chunk_dir(mesh: MeshNode, *, content_id: bytes, chunk_dir: Path) -> None:
    for path in sorted(chunk_dir.glob("chunk_*.holo")):
        chunk_id = _chunk_id_from_name(path.name)
        chunk_bytes = path.read_bytes()

        for peer in mesh.peers:
            for dg in iter_chunk_datagrams(
                content_id,
                int(chunk_id),
                chunk_bytes,
                max_payload=mesh.max_payload,
                auth_key=mesh.auth_key,
                enc_key=mesh.enc_key,
                key_id=int(mesh.enc_key_id),
            ):
                mesh.sock.sendto(dg, peer)
                mesh.counters["sent_datagrams"] += 1
        mesh.counters["sent_chunks"] += 1


def run_send(args: argparse.Namespace) -> int:
    peers = [_parse_addr(p) for p in (args.peer or [])]
    if not peers:
        raise SystemExit("send mode requires at least one --peer ip:port")

    auth_key = bytes.fromhex(args.auth_key) if args.auth_key else None
    enc_key = bytes.fromhex(args.enc_key) if args.enc_key else None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.bind:
        sock.bind(_parse_addr(args.bind))
    sock.setblocking(False)

    mesh = MeshNode(
        sock,
        CortexStore(args.store_root),
        peers=peers,
        max_payload=int(args.max_payload),
        auth_key=auth_key,
        enc_key=enc_key,
        enc_key_id=int(args.enc_key_id),
    )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        img_path = Path(args.image) if args.image else (ROOT / "flower.jpg")
        if not img_path.exists():
            img_path = tmp / "fallback.png"
            _make_fallback_image(img_path)

        wav_path = Path(args.audio) if args.audio else (tmp / "tone.wav")
        if not wav_path.exists():
            _write_tone_wav(wav_path, seconds=float(args.audio_seconds), freq_hz=float(args.tone_hz))

        img_holo = tmp / "img_base.holo"
        aud_holo = tmp / "aud_base.holo"
        encode_image_holo_dir(str(img_path), str(img_holo), target_chunk_kb=int(args.target_chunk_kb))
        encode_audio_holo_dir(str(wav_path), str(aud_holo), target_chunk_kb=int(args.target_chunk_kb))

        t0 = time.time()
        for frame_idx in range(int(args.frames)):
            cid_img = content_id_bytes_from_stream_frame(f"{args.stream_id}/img", frame_idx)
            cid_aud = content_id_bytes_from_stream_frame(f"{args.stream_id}/aud", frame_idx)

            _send_chunk_dir(mesh, content_id=cid_img, chunk_dir=img_holo)
            _send_chunk_dir(mesh, content_id=cid_aud, chunk_dir=aud_holo)

            if args.fps > 0:
                target_t = t0 + (frame_idx + 1) / float(args.fps)
                dt = target_t - time.time()
                if dt > 0:
                    time.sleep(dt)

    print(f"sent chunks={mesh.counters['sent_chunks']} datagrams={mesh.counters['sent_datagrams']}")
    return 0


def run_recv(args: argparse.Namespace) -> int:
    bind = _parse_addr(args.bind)
    auth_key = bytes.fromhex(args.auth_key) if args.auth_key else None
    enc_key = bytes.fromhex(args.enc_key) if args.enc_key else None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(bind)
    sock.setblocking(False)

    store = CortexStore(args.store_root)
    mesh = MeshNode(
        sock,
        store,
        max_payload=int(args.max_payload),
        auth_key=auth_key,
        enc_key=enc_key,
        enc_key_id=int(args.enc_key_id),
    )

    t0 = time.time()
    while time.time() - t0 < float(args.duration):
        mesh.recv_once()
        time.sleep(0.002)

    if args.decode_dir:
        out = Path(args.decode_dir)
        out.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(int(args.frames)):
            cid_img = content_id_bytes_from_stream_frame(f"{args.stream_id}/img", frame_idx)
            cid_aud = content_id_bytes_from_stream_frame(f"{args.stream_id}/aud", frame_idx)

            dir_img = Path(store.content_dir(cid_img))
            dir_aud = Path(store.content_dir(cid_aud))

            if dir_img.exists():
                out_img = out / f"frame_{frame_idx:04d}_img.png"
                try:
                    decode_image_holo_dir(str(dir_img), str(out_img))
                except Exception:
                    pass

            if dir_aud.exists():
                out_wav = out / f"frame_{frame_idx:04d}_aud.wav"
                try:
                    decode_audio_holo_dir(str(dir_aud), str(out_wav))
                except Exception:
                    pass

        print(f"wrote reconstructions under: {out}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stream a tiny scene over MeshNode framing.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_s = sub.add_parser("send")
    ap_s.add_argument("--peer", action="append", default=[])
    ap_s.add_argument("--bind", default=None)
    ap_s.add_argument("--stream-id", required=True)
    ap_s.add_argument("--frames", type=int, default=8)
    ap_s.add_argument("--fps", type=float, default=2.0)
    ap_s.add_argument("--image", default=None)
    ap_s.add_argument("--audio", default=None)
    ap_s.add_argument("--tone-hz", type=float, default=440.0)
    ap_s.add_argument("--audio-seconds", type=float, default=1.0)
    ap_s.add_argument("--target-chunk-kb", type=int, default=16)
    ap_s.add_argument("--store-root", default="cortex_scene_tx")
    ap_s.add_argument("--max-payload", type=int, default=1200)
    ap_s.add_argument("--auth-key", default=None)
    ap_s.add_argument("--enc-key", default=None)
    ap_s.add_argument("--enc-key-id", type=int, default=0)

    ap_r = sub.add_parser("recv")
    ap_r.add_argument("--bind", required=True)
    ap_r.add_argument("--stream-id", required=True)
    ap_r.add_argument("--frames", type=int, default=8)
    ap_r.add_argument("--duration", type=float, default=6.0)
    ap_r.add_argument("--store-root", default="cortex_scene_rx")
    ap_r.add_argument("--decode-dir", default=None)
    ap_r.add_argument("--max-payload", type=int, default=1200)
    ap_r.add_argument("--auth-key", default=None)
    ap_r.add_argument("--enc-key", default=None)
    ap_r.add_argument("--enc-key-id", type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == "send":
        return run_send(args)
    return run_recv(args)


if __name__ == "__main__":
    raise SystemExit(main())
