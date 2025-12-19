"""
holo.tnc.cli

Small helpers to move holo.net datagrams over AFSK WAV files.
"""

from __future__ import annotations

import glob
import json
import os
import wave
from typing import Optional

import numpy as np

from ..net.arch import content_id_bytes_from_uri
from ..net.transport import ChunkAssembler, iter_chunk_datagrams
from ..recovery import REC_HEADER, REC_MAGIC
from .afsk import AFSKModem

_RECOVERY_ID_BASE = 1_000_000


def _ordered_chunk_paths(chunk_dir: str, *, prefer_gain: bool) -> list[str]:
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.holo")))
    if not prefer_gain or not chunk_paths:
        return chunk_paths

    manifest_path = os.path.join(chunk_dir, "manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = None
    if isinstance(data, dict) and "objects" not in data and isinstance(data.get("ordered_chunks"), list):
        ordered = []
        for name in data["ordered_chunks"]:
            path = os.path.join(chunk_dir, str(name))
            if os.path.isfile(path):
                ordered.append(path)
        if ordered:
            return ordered

    scored: list[tuple[float, str]] = []
    for path in chunk_paths:
        meta_path = path + ".meta"
        try:
            with open(meta_path, "r", encoding="ascii") as mf:
                score = float(mf.read().strip())
        except Exception:
            score = 0.0
        scored.append((score, path))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def _chunk_id_from_name(path: str) -> int:
    base = os.path.basename(path)
    if base.startswith("chunk_") and base.endswith(".holo"):
        try:
            return int(base[6:-5])
        except ValueError:
            return 0
    return 0


def _recovery_id_from_name(path: str) -> int:
    base = os.path.basename(path)
    if base.startswith("recovery_") and base.endswith(".holo"):
        try:
            return int(base[9:-5])
        except ValueError:
            return 0
    return 0


def _write_wav(path: str, samples: np.ndarray, fs: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm_i16.tobytes())


def encode_chunk_dir_to_wav(
    chunk_dir: str,
    out_wav: str,
    *,
    uri: str,
    modem: Optional[AFSKModem] = None,
    max_payload: int = 512,
    gap_ms: float = 20.0,
    prefer_gain: bool = False,
    include_recovery: bool = False,
) -> str:
    """
    Encode a .holo directory into an AFSK WAV that carries datagrams.
    """
    if modem is None:
        modem = AFSKModem()
    content_id = content_id_bytes_from_uri(uri)
    chunk_paths = _ordered_chunk_paths(chunk_dir, prefer_gain=prefer_gain)
    if include_recovery:
        chunk_paths = chunk_paths + sorted(glob.glob(os.path.join(chunk_dir, "recovery_*.holo")))

    gap = np.zeros(int(round(float(gap_ms) * 0.001 * float(modem.fs))), dtype=np.float32)
    frames: list[np.ndarray] = []
    for path in chunk_paths:
        base = os.path.basename(path)
        if base.startswith("recovery_"):
            chunk_id = _RECOVERY_ID_BASE + _recovery_id_from_name(path)
        else:
            chunk_id = _chunk_id_from_name(path)
        chunk_bytes = open(path, "rb").read()
        for dg in iter_chunk_datagrams(content_id, int(chunk_id), chunk_bytes, max_payload=int(max_payload)):
            frames.append(modem.encode(dg))
            if gap.size:
                frames.append(gap)

    samples = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    _write_wav(out_wav, samples, int(modem.fs))
    return out_wav


def _chunk_path_from_bytes(out_dir: str, chunk_id: int, chunk_bytes: bytes) -> str:
    if len(chunk_bytes) >= 4 and chunk_bytes[:4] == REC_MAGIC and len(chunk_bytes) >= REC_HEADER.size:
        try:
            (_magic, _rec_ver, _base_kind, _base_codec_ver, _flags,
             _block_count, coded_id, _slice_len, _coeff_len, _payload_len) = REC_HEADER.unpack_from(chunk_bytes, 0)
            return os.path.join(out_dir, f"recovery_{int(coded_id):04d}.holo")
        except Exception:
            pass
    return os.path.join(out_dir, f"chunk_{int(chunk_id):04d}.holo")


def decode_wav_to_chunk_dir(
    wav_path: str,
    out_dir: str,
    *,
    uri: Optional[str] = None,
    modem: Optional[AFSKModem] = None,
    max_partial_age: float = 5.0,
) -> str:
    """
    Decode an AFSK WAV into a .holo directory.
    """
    with wave.open(wav_path, "rb") as wf:
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if modem is None:
        modem = AFSKModem(fs=int(fs))

    wanted_cid = content_id_bytes_from_uri(uri) if uri else None
    assembler = ChunkAssembler(max_partial_age=max_partial_age)
    os.makedirs(out_dir, exist_ok=True)

    if wanted_cid is None:
        out_root = out_dir
    else:
        out_root = out_dir

    for dg in modem.decode(audio):
        res = assembler.push_datagram(dg)
        if res is None:
            continue
        content_id, chunk_id, chunk_bytes = res
        if wanted_cid is not None and content_id != wanted_cid:
            continue
        if wanted_cid is None:
            out_root = os.path.join(out_dir, content_id.hex())
            os.makedirs(out_root, exist_ok=True)
        path = _chunk_path_from_bytes(out_root, int(chunk_id), chunk_bytes)
        with open(path, "wb") as f:
            f.write(chunk_bytes)
    return out_dir
