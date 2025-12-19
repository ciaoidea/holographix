"""
holo.tnc.cli

Small helpers to move holo.net datagrams over AFSK WAV files.
"""

from __future__ import annotations

import glob
import json
import os
import struct
import wave
from typing import Optional

import numpy as np

from ..net.arch import content_id_bytes_from_uri
from ..net.transport import ChunkAssembler, iter_chunk_datagrams
from ..recovery import REC_HEADER, REC_MAGIC
from .afsk import AFSKModem

_RECOVERY_ID_BASE = 1_000_000


def _to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return samples.astype(np.float32, copy=False)
    total = samples.size // int(channels)
    if total <= 0:
        return np.zeros(0, dtype=np.float32)
    use = samples[: total * int(channels)].reshape(total, int(channels))
    return np.mean(use, axis=1).astype(np.float32, copy=False)


def _pcm24_to_float(data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    total = raw.size // 3
    if total <= 0:
        return np.zeros(0, dtype=np.float32)
    use = raw[: total * 3].reshape(total, 3)
    vals = (use[:, 0].astype(np.int32)
            | (use[:, 1].astype(np.int32) << 8)
            | (use[:, 2].astype(np.int32) << 16))
    sign = vals & 0x800000
    vals = vals - (sign << 1)
    return (vals.astype(np.float32) / 8388608.0)


def _decode_wav_bytes(
    data: bytes,
    *,
    audio_format: int,
    channels: int,
    bits_per_sample: int,
) -> np.ndarray:
    fmt = int(audio_format)
    bits = int(bits_per_sample)
    if fmt == 1:  # PCM
        if bits == 8:
            raw = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
            samples = (raw - 128.0) / 128.0
        elif bits == 16:
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 24:
            samples = _pcm24_to_float(data)
        elif bits == 32:
            samples = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM bits per sample: {bits}")
    elif fmt == 3:  # IEEE float
        if bits == 32:
            samples = np.frombuffer(data, dtype="<f4").astype(np.float32, copy=False)
        elif bits == 64:
            samples = np.frombuffer(data, dtype="<f8").astype(np.float32)
        else:
            raise ValueError(f"unsupported float bits per sample: {bits}")
    else:
        raise ValueError(
            f"unsupported WAV format {fmt}; re-encode as PCM 16-bit "
            "(e.g. ffmpeg -i in.wav -ac 1 -ar 48000 -acodec pcm_s16le out.wav)"
        )
    return _to_mono(samples, int(channels))


def _find_data_offset(raw: bytes) -> int:
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return 0
    off = 12
    while off + 8 <= len(raw):
        chunk_id = raw[off:off + 4]
        size = struct.unpack_from("<I", raw, off + 4)[0]
        off += 8
        if chunk_id == b"data":
            return off
        if size > len(raw) - off:
            break
        off += size + (size % 2)
    return 44 if len(raw) >= 44 else 0


def _read_raw_s16le(path: str, *, fs: int, channels: int = 1, skip_bytes: Optional[int] = None) -> tuple[np.ndarray, int]:
    raw = open(path, "rb").read()
    if skip_bytes is None:
        if raw.startswith(b"RIFF"):
            skip_bytes = _find_data_offset(raw)
        else:
            skip_bytes = 0
    if skip_bytes:
        raw = raw[int(skip_bytes):]
    if raw:
        raw = raw[: len(raw) - (len(raw) % 2)]
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return _to_mono(samples, int(channels)), int(fs)


def _resample_linear(audio: np.ndarray, src_fs: int, dst_fs: int) -> np.ndarray:
    src = int(src_fs)
    dst = int(dst_fs)
    if src <= 0 or dst <= 0 or src == dst:
        return audio.astype(np.float32, copy=False)
    n_out = int(round(audio.size * float(dst) / float(src)))
    if n_out <= 0 or audio.size <= 1:
        return np.zeros(max(0, n_out), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _read_wav_fallback(path: str, *, original_error: Optional[Exception] = None) -> tuple[np.ndarray, int]:
    data = open(path, "rb").read()
    try:
        audio_format, channels, sample_rate, bits_per_sample, data_chunk = _parse_wav_chunks(data)
    except Exception:
        msg = "invalid WAV file"
        if original_error:
            msg += f" ({original_error})"
        raise ValueError(msg)

    samples = _decode_wav_bytes(
        data_chunk,
        audio_format=int(audio_format),
        channels=int(channels),
        bits_per_sample=int(bits_per_sample),
    )
    return samples, int(sample_rate)


def _parse_wav_chunks(data: bytes) -> tuple[int, int, int, int, bytes]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("invalid WAV header")

    fmt_chunk: Optional[bytes] = None
    data_chunk: Optional[bytes] = None
    off = 12
    while off + 8 <= len(data):
        chunk_id = data[off:off + 4]
        size = struct.unpack_from("<I", data, off + 4)[0]
        off += 8
        chunk_end = off + int(size)
        actual_end = min(chunk_end, len(data))
        if chunk_id == b"fmt ":
            fmt_chunk = data[off:actual_end]
        elif chunk_id == b"data":
            data_chunk = data[off:actual_end]
            break
        off = chunk_end + (int(size) % 2)

    if fmt_chunk is None or len(fmt_chunk) < 16:
        raise ValueError("WAV fmt chunk missing or too short")
    if data_chunk is None:
        raise ValueError("WAV data chunk missing")

    audio_format, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from(
        "<HHIIHH", fmt_chunk, 0
    )
    if audio_format == 0xFFFE and len(fmt_chunk) >= 40:
        subfmt = fmt_chunk[24:40]
        if len(subfmt) >= 4:
            subtag = struct.unpack_from("<I", subfmt, 0)[0]
            if subtag in (1, 3):
                audio_format = int(subtag)
    return int(audio_format), int(channels), int(sample_rate), int(bits_per_sample), data_chunk


def _read_wav_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        with wave.open(path, "rb") as wf:
            fs = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())
        samples = _decode_wav_bytes(
            raw,
            audio_format=1,
            channels=int(channels),
            bits_per_sample=int(sampwidth) * 8,
        )
        return samples, int(fs)
    except wave.Error as exc:
        return _read_wav_fallback(path, original_error=exc)


def _read_wav_force_pcm16(path: str, *, raw_fs: Optional[int] = None) -> tuple[np.ndarray, int]:
    try:
        return _read_wav_audio(path)
    except Exception:
        data = open(path, "rb").read()
        try:
            _audio_format, channels, sample_rate, _bits, data_chunk = _parse_wav_chunks(data)
            samples = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32) / 32768.0
            return _to_mono(samples, int(channels)), int(sample_rate)
        except Exception:
            if raw_fs is None:
                raise ValueError("cannot force PCM16 without valid WAV header or raw_fs")
            return _read_raw_s16le(path, fs=int(raw_fs))


def fix_wav_to_pcm(
    wav_path: str,
    out_wav: str,
    *,
    fs: Optional[int] = None,
    raw_s16le: bool = False,
    raw_fs: Optional[int] = None,
    raw_channels: int = 1,
    raw_skip: Optional[int] = None,
) -> str:
    """
    Convert a WAV (or raw s16le) into a mono PCM16 WAV.
    """
    if raw_s16le:
        if raw_fs is None:
            raise ValueError("raw_s16le requires raw_fs")
        audio, in_fs = _read_raw_s16le(
            wav_path,
            fs=int(raw_fs),
            channels=int(raw_channels),
            skip_bytes=raw_skip,
        )
    else:
        audio, in_fs = _read_wav_audio(wav_path)

    out_fs = int(fs) if fs is not None else int(in_fs)
    if out_fs != int(in_fs):
        audio = _resample_linear(audio, int(in_fs), out_fs)

    _write_wav(out_wav, audio, out_fs)
    return out_wav


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
    max_chunks: Optional[int] = None,
    fs: Optional[int] = None,
    baud: Optional[int] = None,
    preamble_len: Optional[int] = None,
) -> str:
    """
    Encode a .holo directory into an AFSK WAV that carries datagrams.
    """
    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"chunk_dir not found: {chunk_dir}")
    if modem is None:
        modem_kwargs: dict[str, int] = {}
        if fs is not None:
            modem_kwargs["fs"] = int(fs)
        if baud is not None:
            modem_kwargs["baud"] = int(baud)
        if preamble_len is not None:
            modem_kwargs["preamble_len"] = int(preamble_len)
        modem = AFSKModem(**modem_kwargs)
    content_id = content_id_bytes_from_uri(uri)
    chunk_paths = _ordered_chunk_paths(chunk_dir, prefer_gain=prefer_gain)
    if include_recovery:
        chunk_paths = chunk_paths + sorted(glob.glob(os.path.join(chunk_dir, "recovery_*.holo")))
    if not chunk_paths:
        raise ValueError(f"no chunk_*.holo (or recovery_*.holo) found in {chunk_dir}")
    if max_chunks is not None:
        chunk_paths = chunk_paths[: max(1, int(max_chunks))]

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
    baud: Optional[int] = None,
    preamble_len: Optional[int] = None,
    raw_s16le: bool = False,
    raw_fs: Optional[int] = None,
    raw_channels: int = 1,
    raw_skip: Optional[int] = None,
    force_pcm16: bool = False,
) -> str:
    """
    Decode an AFSK WAV into a .holo directory.
    """
    if raw_s16le:
        if raw_fs is None:
            raise ValueError("raw_s16le requires raw_fs")
        audio, fs = _read_raw_s16le(
            wav_path,
            fs=int(raw_fs),
            channels=int(raw_channels),
            skip_bytes=raw_skip,
        )
    elif force_pcm16:
        audio, fs = _read_wav_force_pcm16(wav_path, raw_fs=raw_fs)
    else:
        audio, fs = _read_wav_audio(wav_path)
    if modem is None:
        modem_kwargs: dict[str, int] = {"fs": int(fs)}
        if baud is not None:
            modem_kwargs["baud"] = int(baud)
        if preamble_len is not None:
            modem_kwargs["preamble_len"] = int(preamble_len)
        modem = AFSKModem(**modem_kwargs)

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
