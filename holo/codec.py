#!/usr/bin/env python3
"""
holo.codec

Genotype layer: chunk formats + versioning + golden interleaving +
single-signal encode/decode primitives (images, audio).

This module deliberately does NOT implement:
- multi-object packing (container.py will)
- storage policies (cortex/* will)
- live reconstruction logic / repair decisions (field.py will)
- networking/transport (net/* will)
"""

from __future__ import annotations

import glob
import math
import os
import struct
import wave
import zlib
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ===================== Constants =====================

MAGIC_IMG = b"HOCH"
MAGIC_AUD = b"HOAU"

VERSION_IMG = 2
VERSION_AUD = 2

PHI = (1.0 + 5.0 ** 0.5) / 2.0

# Header layouts (fixed-size)
# Image header:
#   magic[4], version[u8],
#   H[u32], W[u32], C[u8],
#   B[u32], block_id[u32],
#   coarse_len[u32], resid_len[u32]
IMG_HEADER_SIZE = 30

# Audio header:
#   magic[4], version[u8],
#   channels[u8], sample_width[u8], flags[u8],
#   sample_rate[u32], n_frames[u32],
#   B[u32], block_id[u32],
#   coarse_len[u32],
#   coarse_comp_len[u32], resid_comp_len[u32]
AUD_HEADER_SIZE = 36

# Audio flags
AUD_FLAG_RESID_CLIPPED = 1 << 0

try:
    _BICUBIC = Image.Resampling.BICUBIC  # Pillow >= 9
except AttributeError:  # Pillow < 9
    _BICUBIC = Image.BICUBIC


class ChunkFormatError(ValueError):
    pass


# ===================== Golden permutation =====================

def _golden_step(n: int) -> int:
    """
    Choose an integer step ~ (phi - 1) * n and adjust it so gcd(step, n) == 1.
    Then i -> (i * step) mod n is a full-cycle permutation.
    """
    if n <= 1:
        return 1

    step = max(1, int((PHI - 1.0) * n))
    for _ in range(4096):
        if math.gcd(step, n) == 1:
            return step
        step += 1
    return 1


def _golden_permutation(n: int) -> np.ndarray:
    """perm[i] = (i * step) mod n with coprime step => full-cycle permutation."""
    if n <= 1:
        return np.zeros(max(0, n), dtype=np.int64)
    step = _golden_step(n)
    idx = np.arange(n, dtype=np.int64)
    return (idx * step) % n


def _wipe_old_chunks(out_dir: str) -> None:
    """Remove chunk_*.holo files so old leftovers cannot poison a new encode."""
    for p in glob.glob(os.path.join(out_dir, "chunk_*.holo")):
        try:
            os.remove(p)
        except OSError:
            pass


def _select_block_count(residual_bytes_total: int, coarse_bytes_len: int, target_chunk_kb: int) -> int:
    """
    Roughly choose a block count so that:
      per_chunk_bytes ~= header + coarse + residual_slice

    We assume worst-case residual is incompressible.
    """
    target_bytes = max(256, int(target_chunk_kb) * 1024)
    header_overhead = 96
    overhead = header_overhead + int(coarse_bytes_len)

    if target_bytes <= overhead + 16:
        return 1

    useful = target_bytes - overhead
    return max(1, int(math.ceil(int(residual_bytes_total) / float(useful))))


# ===================== Images =====================

def load_image_rgb_u8(path: str) -> np.ndarray:
    """Load an image and return an RGB uint8 array of shape (H, W, 3)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def save_image_rgb_u8(arr: np.ndarray, path: str) -> None:
    """Save an RGB uint8 array to disk."""
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
    img.save(path)


def encode_image_holo_dir(
    input_path: str,
    out_dir: str,
    *,
    block_count: int = 32,
    coarse_max_side: int = 64,
    target_chunk_kb: Optional[int] = None,
    version: int = VERSION_IMG,
) -> None:
    """
    Encode a single image into a directory of holographic chunks.

    Each chunk contains:
      - a complete coarse thumbnail (PNG bytes)
      - a different slice of the residual (int16), interleaved via golden permutation (v2)

    With all chunks present, reconstruction is pixel-exact (uint8 domain).
    With missing chunks, reconstruction is coarse + partially filled residual.
    """
    if version not in (1, VERSION_IMG):
        raise ValueError(f"Unsupported image codec version: {version}")

    img = load_image_rgb_u8(input_path)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("encode_image_holo_dir expects RGB images (3 channels)")

    max_side = max(h, w)
    scale = min(1.0, float(coarse_max_side) / float(max_side))
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))

    img_pil = Image.fromarray(img, mode="RGB")
    coarse_img = img_pil.resize((cw, ch), resample=_BICUBIC)

    buf = BytesIO()
    coarse_img.save(buf, format="PNG")
    coarse_bytes = buf.getvalue()

    coarse_up = coarse_img.resize((w, h), resample=_BICUBIC)
    coarse_up_arr = np.asarray(coarse_up, dtype=np.uint8)

    residual = img.astype(np.int16) - coarse_up_arr.astype(np.int16)
    residual_flat = residual.reshape(-1)
    residual_bytes_total = residual_flat.size * 2

    if target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_bytes), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), residual_flat.size))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    N = residual_flat.size
    perm = _golden_permutation(N) if (version == VERSION_IMG and block_count > 1) else None

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_flat[block_id::block_count]
        else:
            idx = perm[block_id::block_count]
            vals = residual_flat[idx]

        vals_bytes = vals.astype("<i2", copy=False).tobytes()
        resid_comp = zlib.compress(vals_bytes, level=9)

        header = bytearray()
        header += MAGIC_IMG
        header += struct.pack("B", int(version))
        header += struct.pack(">I", int(h))
        header += struct.pack(">I", int(w))
        header += struct.pack("B", int(c))
        header += struct.pack(">I", int(block_count))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(len(coarse_bytes)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{block_id:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_bytes + resid_comp)


def _parse_image_header(data: bytes) -> Tuple[int, int, int, int, int, int, int, int]:
    if len(data) < IMG_HEADER_SIZE:
        raise ChunkFormatError("Image chunk too small for header")
    off = 0
    if data[off:off + 4] != MAGIC_IMG:
        raise ChunkFormatError("Bad image magic")
    off += 4

    version = data[off]
    off += 1
    if version not in (1, VERSION_IMG):
        raise ChunkFormatError(f"Unsupported image version {version}")

    h = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    w = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    c = data[off]; off += 1
    B = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    block_id = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    coarse_len = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    resid_len = struct.unpack(">I", data[off:off + 4])[0]; off += 4

    return version, h, w, c, B, block_id, coarse_len, resid_len


def _decode_image_holo_core(in_dir: str, *, max_chunks: Optional[int] = None) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    if max_chunks is not None:
        chunk_files = chunk_files[: max(1, int(max_chunks))]

    h = w = c = None
    block_count = None
    version_used = None
    coarse_up_arr = None
    residual_flat = None
    perm = None

    for path in chunk_files:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue

        try:
            version, h_i, w_i, c_i, B_i, block_id, coarse_len, resid_len = _parse_image_header(data)
        except ChunkFormatError:
            continue

        off = IMG_HEADER_SIZE
        if off + coarse_len + resid_len > len(data):
            continue
        if B_i <= 0:
            continue
        if block_id < 0 or block_id >= B_i:
            continue

        coarse_bytes = data[off: off + coarse_len]
        off += coarse_len
        resid_comp = data[off: off + resid_len]

        if residual_flat is None:
            h, w, c = h_i, w_i, c_i
            block_count = B_i
            version_used = version

            try:
                with Image.open(BytesIO(coarse_bytes)) as cim:
                    cim = cim.convert("RGB")
                    coarse_up = cim.resize((w, h), resample=_BICUBIC)
            except Exception:
                continue

            coarse_up_arr = np.asarray(coarse_up, dtype=np.int16)

            residual_flat = np.zeros(h * w * c, dtype=np.int16)
            if version_used == VERSION_IMG and block_count > 1:
                perm = _golden_permutation(residual_flat.size)
        else:
            if (h_i, w_i, c_i, B_i, version) != (h, w, c, block_count, version_used):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if block_count == 1:
            residual_flat[: min(residual_flat.size, vals.size)] = vals[: residual_flat.size]
            continue

        if version_used == VERSION_IMG and perm is not None:
            idx = perm[block_id::block_count]
            n = min(idx.size, vals.size)
            residual_flat[idx[:n]] = vals[:n]
        else:
            pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
            pos = pos[pos < residual_flat.size]
            residual_flat[pos] = vals[: pos.size]

    if residual_flat is None or coarse_up_arr is None:
        raise ValueError(f"No decodable image chunks found in {in_dir}")

    recon = coarse_up_arr + residual_flat.reshape(int(h), int(w), int(c))
    return np.clip(recon, 0, 255).astype(np.uint8)


def decode_image_holo_dir(in_dir: str, output_path: str, *, max_chunks: Optional[int] = None) -> None:
    """Decode an image from a .holo directory and save it."""
    recon = _decode_image_holo_core(in_dir, max_chunks=max_chunks)
    save_image_rgb_u8(recon, output_path)


# ===================== Audio (WAV) =====================

def _read_wav_as_int16(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Read PCM WAV and return (samples_int16, sample_rate, channels).

    Supports 16-bit PCM and 24-bit PCM (24-bit is down-converted to int16).
    samples_int16 shape: (n_frames, n_channels).
    """
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    total = n_frames * ch

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=False)
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size != total * 3:
            raise ValueError("Invalid 24-bit WAV payload size")
        b = b.reshape(-1, 3)
        vals = (b[:, 0].astype(np.int32) |
                (b[:, 1].astype(np.int32) << 8) |
                (b[:, 2].astype(np.int32) << 16))
        mask = 1 << 23
        vals = (vals ^ mask) - mask
        data = (vals >> 8).astype(np.int16)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    return data.reshape(-1, ch), int(sr), int(ch)


def _write_wav_int16(path: str, audio: np.ndarray, sr: int) -> None:
    """Write int16 PCM WAV. audio shape: (n_frames, n_channels)."""
    audio = np.asarray(audio, dtype=np.int16)
    n_frames, ch = audio.shape
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(ch))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio.astype("<i2", copy=False).tobytes())


def encode_audio_holo_dir(
    input_wav: str,
    out_dir: str,
    *,
    block_count: int = 16,
    coarse_max_frames: int = 2048,
    target_chunk_kb: Optional[int] = None,
    version: int = VERSION_AUD,
) -> None:
    """
    Encode a WAV file into a directory of holographic chunks.

    Each chunk contains:
      - a complete coarse track (zlib-compressed int16 frames)
      - a different slice of the residual (int16), interleaved via golden permutation (v2)

    Reconstruction with all chunks present is exact *when the residual fits in int16*.
    If some residual amplitudes exceed int16, they are clipped (flagged in the header)
    to prevent wrap-around; this trades bit-exactness for stability.
    """
    if version not in (1, VERSION_AUD):
        raise ValueError(f"Unsupported audio codec version: {version}")

    audio, sr, ch = _read_wav_as_int16(input_wav)
    n_frames = int(audio.shape[0])
    if n_frames < 2:
        raise ValueError("Audio too short to encode")

    coarse_len = int(min(max(2, int(coarse_max_frames)), n_frames))
    idx = np.linspace(0, n_frames - 1, coarse_len, dtype=np.int64)
    coarse = audio[idx]

    t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
    k0 = np.floor(t).astype(np.int64)
    k1 = np.clip(k0 + 1, 0, coarse_len - 1)
    alpha = (t - k0).astype(np.float64)[:, None]
    coarse_f = coarse.astype(np.float64)
    coarse_up = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
    coarse_up = np.round(coarse_up).astype(np.int16)

    diff32 = audio.astype(np.int32) - coarse_up.astype(np.int32)
    clipped = bool((diff32 < -32768).any() or (diff32 > 32767).any())
    residual = np.clip(diff32, -32768, 32767).astype(np.int16)

    residual_flat = residual.reshape(-1)
    residual_bytes_total = residual_flat.size * 2

    coarse_comp = zlib.compress(coarse.astype("<i2", copy=False).tobytes(), level=9)

    if target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_comp), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), residual_flat.size))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    N = residual_flat.size
    perm = _golden_permutation(N) if (version == VERSION_AUD and block_count > 1) else None

    flags = 0
    if clipped:
        flags |= AUD_FLAG_RESID_CLIPPED

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_flat[block_id::block_count]
        else:
            idxb = perm[block_id::block_count]
            vals = residual_flat[idxb]

        resid_comp = zlib.compress(vals.astype("<i2", copy=False).tobytes(), level=9)

        header = bytearray()
        header += MAGIC_AUD
        header += struct.pack("B", int(version))
        header += struct.pack("B", int(ch))
        header += struct.pack("B", 2)             # internal sample width (int16)
        header += struct.pack("B", int(flags))    # flags
        header += struct.pack(">I", int(sr))
        header += struct.pack(">I", int(n_frames))
        header += struct.pack(">I", int(block_count))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(coarse_len))
        header += struct.pack(">I", int(len(coarse_comp)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{block_id:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_comp + resid_comp)


def _parse_audio_header(data: bytes) -> Tuple[int, int, int, int, int, int, int, int, int, int, int]:
    if len(data) < AUD_HEADER_SIZE:
        raise ChunkFormatError("Audio chunk too small for header")
    off = 0
    if data[off:off + 4] != MAGIC_AUD:
        raise ChunkFormatError("Bad audio magic")
    off += 4

    version = data[off]; off += 1
    if version not in (1, VERSION_AUD):
        raise ChunkFormatError(f"Unsupported audio version {version}")

    ch = data[off]; off += 1
    sampwidth = data[off]; off += 1
    flags = data[off]; off += 1
    sr = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    n_frames = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    B = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    block_id = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    coarse_len = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    coarse_size = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    resid_size = struct.unpack(">I", data[off:off + 4])[0]; off += 4

    return version, ch, sampwidth, flags, sr, n_frames, B, block_id, coarse_len, coarse_size, resid_size


def decode_audio_holo_dir(in_dir: str, output_wav: str, *, max_chunks: Optional[int] = None) -> None:
    """Decode a WAV file from a .holo directory and write a 16-bit PCM WAV."""
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")
    if max_chunks is not None:
        chunk_files = chunk_files[: max(1, int(max_chunks))]

    sr = ch = n_frames = None
    block_count = None
    coarse_len = None
    version_used = None

    coarse_up = None
    residual_flat = None
    perm = None

    for path in chunk_files:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue

        try:
            (version, ch_i, sampwidth, flags, sr_i, n_frames_i,
             B_i, block_id, coarse_len_i, coarse_size, resid_size) = _parse_audio_header(data)
        except ChunkFormatError:
            continue

        if sampwidth != 2:
            continue
        if B_i <= 0:
            continue
        if block_id < 0 or block_id >= B_i:
            continue

        off = AUD_HEADER_SIZE
        if off + coarse_size + resid_size > len(data):
            continue

        coarse_comp = data[off:off + coarse_size]
        off += coarse_size
        resid_comp = data[off:off + resid_size]

        if residual_flat is None:
            sr, ch, n_frames = int(sr_i), int(ch_i), int(n_frames_i)
            block_count, coarse_len = int(B_i), int(coarse_len_i)
            version_used = int(version)

            try:
                coarse_bytes = zlib.decompress(coarse_comp)
                coarse = np.frombuffer(coarse_bytes, dtype="<i2").astype(np.int16, copy=False)
                coarse = coarse.reshape(coarse_len, ch)
            except Exception:
                continue

            t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
            k0 = np.floor(t).astype(np.int64)
            k1 = np.clip(k0 + 1, 0, coarse_len - 1)
            alpha = (t - k0).astype(np.float64)[:, None]
            coarse_f = coarse.astype(np.float64)
            coarse_up = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
            coarse_up = np.round(coarse_up).astype(np.int16)

            residual_flat = np.zeros(n_frames * ch, dtype=np.int16)

            if version_used == VERSION_AUD and block_count > 1:
                perm = _golden_permutation(residual_flat.size)
        else:
            if (sr_i, ch_i, n_frames_i, B_i, coarse_len_i, version) != (sr, ch, n_frames, block_count, coarse_len, version_used):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if block_count == 1:
            residual_flat[: min(residual_flat.size, vals.size)] = vals[: residual_flat.size]
            continue

        if version_used == VERSION_AUD and perm is not None:
            idxb = perm[block_id::block_count]
            n = min(idxb.size, vals.size)
            residual_flat[idxb[:n]] = vals[:n]
        else:
            pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
            pos = pos[pos < residual_flat.size]
            residual_flat[pos] = vals[: pos.size]

    if residual_flat is None or coarse_up is None:
        raise ValueError(f"No decodable audio chunks found in {in_dir}")

    residual = residual_flat.reshape(n_frames, ch)
    recon = coarse_up.astype(np.int32) + residual.astype(np.int32)
    recon = np.clip(recon, -32768, 32767).astype(np.int16)
    _write_wav_int16(output_wav, recon, int(sr))


# ===================== Mode detection =====================

def detect_mode_from_extension(path: str) -> str:
    """Return 'image' or 'audio' based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return "audio"
    return "image"


def detect_mode_from_chunk_dir(in_dir: str) -> str:
    """Return 'image' or 'audio' by reading the first chunk magic bytes."""
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")
    with open(chunk_files[0], "rb") as f:
        magic = f.read(4)
    if magic == MAGIC_IMG:
        return "image"
    if magic == MAGIC_AUD:
        return "audio"
    raise ValueError("Unknown chunk magic (not image/audio)")

