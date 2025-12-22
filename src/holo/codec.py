#!/usr/bin/env python3
"""
holo.codec

Genotype layer: chunk formats, versioning, golden interleaving and
single-signal encode/decode primitives for images and audio.

This module is intentionally low-level and *stateless*.
It defines how a single image or a single audio track is mapped to a set
of holographic chunks, and how any subset of those chunks can be used
to reconstruct a coherent percept again.

It does NOT implement:
  - multi-object packing (see container.py in the full project)
  - storage policies or aging (see cortex/*)
  - live reconstruction / healing logic (see field.py)
  - networking / transport (see net/*)

Typical usage at this level looks like:

    from holo.codec import (
        encode_image_holo_dir,
        decode_image_holo_dir,
        encode_audio_holo_dir,
        decode_audio_holo_dir,
        stack_image_holo_dirs,
    )

    # Encode a single RGB image into holographic chunks
    encode_image_holo_dir("frame.png", "frame.png.holo", target_chunk_kb=32)

    # Decode from chunks back to an image
    decode_image_holo_dir("frame.png.holo", "frame_recon.png")

    # Stack several holographic images (same scene) to improve SNR over time
    stack_image_holo_dirs(
        ["t0.png.holo", "t1.png.holo", "t2.png.holo"],
        "frame_stacked.png",
    )

    # Encode / decode audio (16-bit PCM WAV)
    encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
    decode_audio_holo_dir("track.wav.holo", "track_recon.wav")
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import math
import os
import struct
import wave
import zlib
from io import BytesIO
from typing import Optional, Tuple, Sequence

import numpy as np
from PIL import Image

from .recovery import (
    REC_KIND_AUDIO,
    REC_KIND_IMAGE,
    build_recovery_chunks,
    parse_recovery_chunk,
    recover_missing_slices,
)
from .models.coarse import get_coarse_model

# ===================== Constants and basic types =====================

# Magic identifiers for chunk type (first four bytes of every chunk)
MAGIC_IMG = b"HOCH"  # HOlographic CHunk (image)
MAGIC_AUD = b"HOAU"  # HOlographic AUdio

# Codec versions. Versioning sits in the header of each chunk.
# Version 1 used a simple strided layout for residual sampling.
# Version 2 uses golden-ratio interleaving of the residual array.
VERSION_IMG = 2
VERSION_AUD = 2
VERSION_IMG_OLO = 3
VERSION_AUD_OLO = 3

# Golden ratio constant used for the permutation step.
PHI = (1.0 + 5.0 ** 0.5) / 2.0

# Fixed-size header layouts.
# Image header:
#   magic[4], version[u8],
#   H[u32], W[u32], C[u8],
#   B[u32], block_id[u32],
#   coarse_len[u32], resid_len[u32]
IMG_HEADER_SIZE = 30
OLOI_META = struct.Struct(">4sBBBBHHI")
OLOI_META_V2 = struct.Struct(">4sBBBBHHIB")
OLOI_MAGIC = b"OLOI"
OLOI_FLAG_COEFF_CLIPPED = 1 << 0

# Audio header:
#   magic[4], version[u8],
#   channels[u8], sample_width[u8], flags[u8],
#   sample_rate[u32], n_frames[u32],
#   B[u32], block_id[u32],
#   coarse_len[u32],
#   coarse_comp_len[u32], resid_comp_len[u32]
AUD_HEADER_SIZE = 36
OLOA_META = struct.Struct(">4sBBBBII")
OLOA_MAGIC = b"OLOA"
OLOA_FLAG_COEFF_CLIPPED = 1 << 0

# Audio flags: placed in the header to indicate extra conditions.
AUD_FLAG_RESID_CLIPPED = 1 << 0  # residual had values outside int16 and was clipped


try:
    # Pillow >= 9
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    # Older Pillow
    _BICUBIC = Image.BICUBIC


class ChunkFormatError(ValueError):
    """Raised when a chunk header or payload does not match the expected format."""


# ===================== Golden permutation (residual interleaving) =====================

def _golden_step(n: int) -> int:
    """
    Compute an integer step that is approximately (phi - 1) * n and coprime with n.

    The idea is to walk the residual array with a "rotation" that is as
    incommensurate as possible with its length. Using a step close to n/phi
    and forcing gcd(step, n) == 1 ensures that the mapping

        i -> (i * step) mod n

    visits every position exactly once (a full cycle).

    Parameters
    ----------
    n : int
        Length of the residual array.

    Returns
    -------
    int
        A step size that is coprime with n.
    """
    if n <= 1:
        return 1

    step = max(1, int((PHI - 1.0) * n))

    # Small linear search to find a coprime step.
    for _ in range(4096):
        if math.gcd(step, n) == 1:
            return step
        step += 1

    # Worst case fallback, should almost never be hit.
    return 1


def _golden_permutation(n: int) -> np.ndarray:
    """
    Build a full-cycle permutation of indices 0..n-1 using a golden-ratio step.

    perm[i] = (i * step) mod n

    with step chosen so that gcd(step, n) = 1.

    This permutation is used to "spread" the residual over chunks so that
    each chunk samples the entire signal, not a contiguous block.

    Parameters
    ----------
    n : int
        Length of the residual.

    Returns
    -------
    np.ndarray
        Array of shape (n,) with dtype int64, containing the permuted indices.
    """
    if n <= 1:
        return np.zeros(max(0, n), dtype=np.int64)

    step = _golden_step(n)
    idx = np.arange(n, dtype=np.int64)
    return (idx * step) % n


def _wipe_old_chunks(out_dir: str) -> None:
    """
    Remove existing chunk_*.holo and recovery_*.holo files from a directory.

    This is used before a new encode so that stale chunks from a previous
    encode do not accidentally "mix" into a new holographic field.
    """
    patterns = ("chunk_*.holo", "recovery_*.holo")
    for pat in patterns:
        for p in glob.glob(os.path.join(out_dir, pat)):
            try:
                os.remove(p)
            except OSError:
                # If deletion fails we silently ignore; worst case a stale chunk remains.
                pass


def _select_block_count(residual_bytes_total: int, coarse_bytes_len: int, target_chunk_kb: int) -> int:
    """
    Approximate a reasonable number of chunks given a target chunk size.

    The goal is to keep each chunk roughly around target_chunk_kb, once you
    account for header + coarse representation + its share of residual.

    We assume residual is incompressible in the worst case so that the
    estimate is conservative.

    Parameters
    ----------
    residual_bytes_total : int
        Total size of the residual vector in bytes (before compression).
    coarse_bytes_len : int
        Size in bytes of the coarse representation for this signal.
    target_chunk_kb : int
        Target chunk size in kilobytes.

    Returns
    -------
    int
        Suggested number of chunks (block_count).
    """
    target_bytes = max(256, int(target_chunk_kb) * 1024)
    header_overhead = 96  # generous margin for headers and framing
    overhead = header_overhead + int(coarse_bytes_len)

    # If the requested size is too small to even hold overhead, fall back to 1 chunk.
    if target_bytes <= overhead + 16:
        return 1

    useful = target_bytes - overhead
    return max(1, int(math.ceil(int(residual_bytes_total) / float(useful))))


def _select_block_count_bytes(residual_bytes_total: int, coarse_bytes_len: int, target_chunk_bytes: int) -> int:
    """
    Choose block_count to keep chunks under target_chunk_bytes (roughly).

    This uses the same conservative estimate as _select_block_count but with
    a byte budget instead of kilobytes.
    """
    target_bytes = max(256, int(target_chunk_bytes))
    overhead = IMG_HEADER_SIZE + int(coarse_bytes_len)

    if target_bytes <= overhead + 16:
        return 1

    useful = target_bytes - overhead
    return max(1, int(math.ceil(int(residual_bytes_total) / float(useful))))


def _residual_slice_lengths(total_len: int, block_count: int, elem_bytes: int) -> Tuple[list[int], int]:
    lengths: list[int] = []
    for block_id in range(int(block_count)):
        if block_id >= total_len:
            n = 0
        else:
            n = (int(total_len) - block_id + int(block_count) - 1) // int(block_count)
        lengths.append(int(n) * int(elem_bytes))
    max_len = max(lengths) if lengths else 0
    return lengths, int(max_len)


def _apply_residual_slice(
    coeff_vec: np.ndarray,
    block_id: int,
    block_count: int,
    vals: np.ndarray,
    perm: Optional[np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> None:
    if block_count == 1 or perm is None:
        pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
        pos = pos[pos < coeff_vec.size]
        coeff_vec[pos] = vals[: pos.size]
        if mask is not None:
            mask[pos] = True
    else:
        idx = perm[block_id::block_count]
        n = min(idx.size, vals.size)
        coeff_vec[idx[:n]] = vals[:n]
        if mask is not None:
            mask[idx[:n]] = True


def _read_chunk_score(meta_path: str) -> Optional[float]:
    try:
        with open(meta_path, "r", encoding="ascii") as f:
            raw = f.read().strip()
        return float(raw)
    except Exception:
        return None


def _write_chunk_manifest(
    out_dir: str,
    *,
    base_kind: str,
    codec_version: int,
    block_count: int,
    entries: Sequence[dict],
) -> None:
    manifest = {
        "kind": "chunk_manifest",
        "manifest_version": 1,
        "base_kind": str(base_kind),
        "codec_version": int(codec_version),
        "block_count": int(block_count),
        "ordered_chunks": [e["file"] for e in entries],
        "chunks": entries,
    }
    path = os.path.join(out_dir, "manifest.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except OSError:
        pass


def _load_chunk_manifest(in_dir: str) -> Optional[dict]:
    path = os.path.join(in_dir, "manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "objects" in data:
        return None
    return data


def _select_chunk_files(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
) -> list[str]:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        return []
    if not prefer_gain:
        if max_chunks is not None:
            return chunk_files[: max(1, int(max_chunks))]
        return chunk_files

    ordered: Optional[list[str]] = None
    manifest = _load_chunk_manifest(in_dir)
    if manifest and isinstance(manifest.get("ordered_chunks"), list):
        ordered = []
        for name in manifest["ordered_chunks"]:
            path = os.path.join(in_dir, str(name))
            if os.path.isfile(path):
                ordered.append(path)
        if not ordered:
            ordered = None

    if ordered is None:
        scored = []
        for path in chunk_files:
            score = _read_chunk_score(path + ".meta")
            scored.append((float(score) if score is not None else 0.0, path))
        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [p for _, p in scored]

    if max_chunks is not None:
        ordered = ordered[: max(1, int(max_chunks))]
    return ordered


# ===================== Image helpers and codec =====================

def load_image_rgb_u8(path: str) -> np.ndarray:
    """
    Load an image from disk as RGB uint8.

    Parameters
    ----------
    path : str
        Path to the input image file.

    Returns
    -------
    np.ndarray
        Array with shape (H, W, 3) and dtype uint8.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)


def save_image_rgb_u8(arr: np.ndarray, path: str) -> None:
    """
    Save an RGB uint8 array to disk.

    Parameters
    ----------
    arr : np.ndarray
        Array with shape (H, W, 3) and dtype uint8.
    path : str
        Target output path.
    """
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
    img.save(path)


_DCT_CACHE: dict[int, np.ndarray] = {}
_ZIGZAG_CACHE: dict[int, np.ndarray] = {}
_JPEG_QUANT_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


def _get_dct_matrix(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Block size must be positive")
    m = _DCT_CACHE.get(int(n))
    if m is not None:
        return m

    k = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(n, dtype=np.float64)[None, :]
    mat = np.sqrt(2.0 / float(n)) * np.cos(np.pi * (j + 0.5) * k / float(n))
    mat[0, :] = 1.0 / np.sqrt(float(n))
    m = mat.astype(np.float32)
    _DCT_CACHE[int(n)] = m
    return m


def _zigzag_indices(n: int) -> np.ndarray:
    cached = _ZIGZAG_CACHE.get(int(n))
    if cached is not None:
        return cached

    coords = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c < n:
                coords.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r < n:
                coords.append((r, c))
                r += 1
                c -= 1
    arr = np.array(coords, dtype=np.int64)
    _ZIGZAG_CACHE[int(n)] = arr
    return arr


def _scaled_quant_table(block: int, quality: int) -> np.ndarray:
    q = int(np.clip(quality, 1, 100))
    base = _JPEG_QUANT_LUMA
    if q < 50:
        scale = 5000.0 / float(q)
    else:
        scale = 200.0 - 2.0 * float(q)
    scaled8 = np.floor((base * scale + 50.0) / 100.0)
    scaled8 = np.clip(scaled8, 1.0, 32767.0)
    if block == 8:
        return scaled8.astype(np.float32)

    # Simple bilinear resize of the base table to match the requested block size.
    coords = np.linspace(0.0, base.shape[0] - 1.0, int(block))
    tmp = np.empty((int(block), base.shape[1]), dtype=np.float32)
    for j in range(base.shape[1]):
        tmp[:, j] = np.interp(coords, np.arange(base.shape[0]), base[:, j])
    out = np.empty((int(block), int(block)), dtype=np.float32)
    for i in range(int(block)):
        out[i, :] = np.interp(coords, np.arange(base.shape[1]), tmp[i, :])
    return np.clip(np.round(out), 1.0, 32767.0).astype(np.float32)


def encode_image_holo_dir(
    input_path: str,
    out_dir: str,
    *,
    block_count: int = 32,
    coarse_max_side: int = 16,
    target_chunk_kb: Optional[int] = None,
    max_chunk_bytes: Optional[int] = None,
    coarse_format: str = "PNG",
    coarse_quality: Optional[int] = None,
    version: int = VERSION_IMG,
) -> None:
    """
    Encode a single RGB image into a directory of holographic chunks.

    Conceptually the codec does two things:
      1. Builds a coarse thumbnail of the image, resizes it back to full size.
      2. Computes a residual (original - coarse_up) in int16 and distributes it
         across `block_count` chunks using a golden-ratio permutation.

    Each chunk carries:
      - the full coarse thumbnail as PNG bytes,
      - a different slice of the residual, compressed with zlib.

    With all chunks present, reconstruction is pixel-exact in uint8.
    With only a subset, reconstruction remains globally coherent but with
    reduced detail (the residual has missing entries set to zero).

    Parameters
    ----------
    input_path : str
        Path to the input image (any format Pillow can read).
    out_dir : str
        Directory to create or reuse for chunks (e.g. "image.png.holo").
    block_count : int, optional
        Desired number of residual chunks. Ignored if target_chunk_kb is set,
        because in that case the codec will compute a block_count that matches
        the requested approximate chunk size.
    coarse_max_side : int, optional
        Maximum side length of the coarse thumbnail. Larger values give a more
        detailed coarse base and slightly lower compression; smaller values
        give a blurrier base and more work for the residual.
    target_chunk_kb : int, optional
        Approximate chunk size in kilobytes. If provided, overrides
        block_count with an automatically chosen number of chunks.
    max_chunk_bytes : int, optional
        Harder cap on chunk size in bytes (approximate, conservative). If set,
        it overrides target_chunk_kb and block_count to aim for packet-sized
        chunks. Useful when you want 1 UDP datagram = 1 chunk.
    version : int, optional
        Codec version. Currently 1 (legacy) or 2 (golden interleaving).

    Example
    -------
    Encode an image into ~32 KB chunks:

        encode_image_holo_dir(
            "frame.png",
            "frame.png.holo",
            target_chunk_kb=32,
        )
    """
    if version not in (1, VERSION_IMG):
        raise ValueError(f"Unsupported image codec version: {version}")

    img = load_image_rgb_u8(input_path)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("encode_image_holo_dir expects RGB images with 3 channels")

    # Build a coarse thumbnail by downscaling the largest side to coarse_max_side
    max_side = max(h, w)
    scale = min(1.0, float(coarse_max_side) / float(max_side))
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))

    img_pil = Image.fromarray(img, mode="RGB")
    coarse_img = img_pil.resize((cw, ch), resample=_BICUBIC)

    # Serialize the coarse thumbnail as PNG
    buf = BytesIO()
    save_kwargs = {"format": str(coarse_format)}
    if coarse_quality is not None:
        save_kwargs["quality"] = int(coarse_quality)
    coarse_img.save(buf, **save_kwargs)
    coarse_bytes = buf.getvalue()

    # Upsample coarse back to full resolution and compute residual
    coarse_up = coarse_img.resize((w, h), resample=_BICUBIC)
    coarse_up_arr = np.asarray(coarse_up, dtype=np.uint8)

    residual = img.astype(np.int16) - coarse_up_arr.astype(np.int16)
    residual_flat = residual.reshape(-1)
    residual_bytes_total = residual_flat.size * 2  # int16 => 2 bytes per element

    # If a chunk size target is given, pick block_count automatically
    if max_chunk_bytes is not None:
        block_count = _select_block_count_bytes(residual_bytes_total, len(coarse_bytes), int(max_chunk_bytes))
    elif target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_bytes), int(target_chunk_kb))

    # Guard against degenerate cases where residual has fewer elements than block_count
    block_count = max(1, min(int(block_count), residual_flat.size))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    # Build the golden permutation once for this signal
    N = residual_flat.size
    perm = _golden_permutation(N) if (version == VERSION_IMG and block_count > 1) else None

    # Emit each chunk
    entries: list[dict] = []
    for block_id in range(block_count):
        if perm is None or block_count == 1:
            # Legacy layout: simple strided sampling of the residual
            vals = residual_flat[block_id::block_count]
        else:
            # Golden layout: use the permuted indices
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

        # Store a tiny metadata sidecar with a simple perceptual gain metric
        # (L1 magnitude of the residual slice). The gossip loop can use this
        # to prioritize higher-impact chunks.
        gain = float(np.sum(np.abs(vals), dtype=np.float64))
        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{gain:.6f}")
        except OSError:
            pass
        entries.append(
            {"file": os.path.basename(chunk_path), "block_id": int(block_id), "score": float(gain)}
        )

    if entries:
        entries.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        _write_chunk_manifest(
            out_dir,
            base_kind="image",
            codec_version=int(version),
            block_count=int(block_count),
            entries=entries,
        )


def encode_image_olonomic_holo_dir(
    input_path: str,
    out_dir: str,
    *,
    block_count: int = 32,
    coarse_max_side: int = 16,
    coarse_model: str = "downsample",
    target_chunk_kb: Optional[int] = None,
    max_chunk_bytes: Optional[int] = None,
    quality: int = 50,
    dct_block: int = 8,
    recovery: Optional[str] = None,
    overhead: float = 0.0,
    recovery_seed: Optional[int] = None,
    version: int = VERSION_IMG_OLO,
) -> None:
    if version != VERSION_IMG_OLO:
        raise ValueError(f"Unsupported olonomic image codec version: {version}")

    img = load_image_rgb_u8(input_path)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("encode_image_olonomic_holo_dir expects RGB images with 3 channels")

    block = int(dct_block)
    if block <= 0:
        raise ValueError("dct_block must be positive")
    quality_u8 = int(np.clip(quality, 1, 100))

    model = get_coarse_model(coarse_model, kind="image")
    model_name = model.name
    name_bytes = model_name.encode("ascii", errors="ignore") or b"downsample"
    if len(name_bytes) > 255:
        name_bytes = name_bytes[:255]

    coarse_payload_bytes = model.encode(
        img,
        coarse_max_side=int(coarse_max_side),
        block_size=int(block),
        quality=int(quality_u8),
    )
    coarse_up = model.decode(
        coarse_payload_bytes,
        target_shape=(int(h), int(w), int(c)),
        block_size=int(block),
        quality=int(quality_u8),
    )
    coarse_up_arr = np.asarray(coarse_up, dtype=np.int16, order="C")

    residual = img.astype(np.int16) - coarse_up_arr
    residual_f = residual.astype(np.float32, copy=False)

    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block
    residual_f = np.pad(residual_f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    T = _get_dct_matrix(block)
    quant = _scaled_quant_table(block, quality_u8)
    zigzag = _zigzag_indices(block)
    zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]

    blocks_h = residual_f.shape[0] // block
    blocks_w = residual_f.shape[1] // block

    coeff_parts: list[np.ndarray] = []
    clipped = False

    for ch_idx in range(c):
        plane = residual_f[:, :, ch_idx]
        for by in range(blocks_h):
            y0 = by * block
            y1 = y0 + block
            for bx in range(blocks_w):
                x0 = bx * block
                x1 = x0 + block
                block_vals = plane[y0:y1, x0:x1]
                coeff = T @ block_vals @ T.T
                q = np.rint(coeff / quant)
                q_clip = np.clip(q, -32768.0, 32767.0)
                if not clipped and np.any(q != q_clip):
                    clipped = True
                coeff_parts.append(q_clip[zz_r, zz_c].astype(np.int16, copy=False))

    residual_coeff = np.concatenate(coeff_parts, axis=0) if coeff_parts else np.zeros(0, dtype=np.int16)
    residual_bytes_total = residual_coeff.size * 2

    coarse_payload = OLOI_META_V2.pack(
        OLOI_MAGIC,
        2,  # meta version
        int(block),
        quality_u8,
        OLOI_FLAG_COEFF_CLIPPED if clipped else 0,
        int(pad_h),
        int(pad_w),
        int(len(coarse_payload_bytes)),
        int(len(name_bytes)),
    ) + name_bytes + coarse_payload_bytes

    if max_chunk_bytes is not None:
        block_count = _select_block_count_bytes(residual_bytes_total, len(coarse_payload), int(max_chunk_bytes))
    elif target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), max(1, residual_coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(residual_coeff.size) if block_count > 1 else None
    use_recovery = bool(recovery) and str(recovery).lower() == "rlnc" and float(overhead) > 0.0
    slice_bytes = [b""] * block_count if use_recovery else None
    entries: list[dict] = []

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_coeff[block_id::block_count]
        else:
            idx = perm[block_id::block_count]
            vals = residual_coeff[idx]

        vals_bytes = vals.astype("<i2", copy=False).tobytes()
        if slice_bytes is not None:
            slice_bytes[block_id] = vals_bytes
        resid_comp = zlib.compress(vals_bytes, level=9)

        header = bytearray()
        header += MAGIC_IMG
        header += struct.pack("B", int(version))
        header += struct.pack(">I", int(h))
        header += struct.pack(">I", int(w))
        header += struct.pack("B", int(c))
        header += struct.pack(">I", int(block_count))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(len(coarse_payload)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{block_id:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_payload + resid_comp)

        score = float(np.sum(vals.astype(np.float64) ** 2))
        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{score:.6f}")
        except OSError:
            pass
        entries.append(
            {"file": os.path.basename(chunk_path), "block_id": int(block_id), "score": float(score)}
        )

    if slice_bytes is not None:
        recovery_chunks = build_recovery_chunks(
            slice_bytes,
            base_kind=REC_KIND_IMAGE,
            base_codec_version=int(version),
            overhead=float(overhead),
            seed=recovery_seed,
        )
        for idx, chunk_bytes in enumerate(recovery_chunks):
            rec_path = os.path.join(out_dir, f"recovery_{idx:04d}.holo")
            with open(rec_path, "wb") as f:
                f.write(chunk_bytes)

    if entries:
        entries.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        _write_chunk_manifest(
            out_dir,
            base_kind="image",
            codec_version=int(version),
            block_count=int(block_count),
            entries=entries,
        )


def _parse_image_header(data: bytes) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Parse the fixed-size image header from a chunk payload.

    Returns a tuple:
      (version, H, W, C, block_count, block_id, coarse_len, resid_len)
    """
    if len(data) < IMG_HEADER_SIZE:
        raise ChunkFormatError("Image chunk too small for header")

    off = 0
    if data[off:off + 4] != MAGIC_IMG:
        raise ChunkFormatError("Bad image magic")
    off += 4

    version = data[off]
    off += 1
    if version not in (1, VERSION_IMG, VERSION_IMG_OLO):
        raise ChunkFormatError(f"Unsupported image version {version}")

    h = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    w = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    c = data[off]; off += 1
    B = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    block_id = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    coarse_len = struct.unpack(">I", data[off:off + 4])[0]; off += 4
    resid_len = struct.unpack(">I", data[off:off + 4])[0]; off += 4

    return version, h, w, c, B, block_id, coarse_len, resid_len


def _decode_image_holo_core_v1_v2(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    return_mask: bool = False,
) -> np.ndarray:
    """
    Internal decoder for image versions 1/2 that reconstructs an RGB image from a .holo directory.

    It reads up to max_chunks chunks (if given), or all available chunks otherwise.
    For each chunk it:
      - parses the header,
      - reconstructs / reuses the coarse thumbnail,
      - decompresses and places its residual slice into the global residual vector.

    This function returns an RGB uint8 array and does not write to disk.
    """
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    h = w = c = None
    block_count = None
    version_used = None
    coarse_up_arr = None
    residual_flat = None
    mask_flat = None
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

        if version == VERSION_IMG_OLO:
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
            # First valid chunk sets global geometry and allocates buffers
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
            if return_mask:
                mask_flat = np.zeros(h * w * c, dtype=bool)

            if version_used == VERSION_IMG and block_count > 1:
                perm = _golden_permutation(residual_flat.size)
        else:
            # Discard chunks whose headers do not match the first one
            if (h_i, w_i, c_i, B_i, version) != (h, w, c, block_count, version_used):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        _apply_residual_slice(
            residual_flat,
            int(block_id),
            int(block_count),
            vals,
            perm,
            mask_flat if return_mask else None,
        )

    if residual_flat is None or coarse_up_arr is None:
        raise ValueError(f"No decodable image chunks found in {in_dir}")

    recon = coarse_up_arr + residual_flat.reshape(int(h), int(w), int(c))
    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)

    if return_mask and mask_flat is not None:
        block = 8
        mask_img = mask_flat.reshape(int(h), int(w), int(c)).astype(np.float32)
        mask_gray = np.mean(mask_img, axis=2)
        pad_h = (block - (int(h) % block)) % block
        pad_w = (block - (int(w) % block)) % block
        mask_pad = np.pad(mask_gray, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        blocks_h = mask_pad.shape[0] // block
        blocks_w = mask_pad.shape[1] // block
        conf = mask_pad.reshape(blocks_h, block, blocks_w, block).mean(axis=(1, 3))
        return recon_u8, conf

    return recon_u8


def _decode_image_holo_core_v3(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = False,
) -> np.ndarray:
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    recovery_files = []
    if use_recovery is not False:
        recovery_files = sorted(glob.glob(os.path.join(in_dir, "recovery_*.holo")))
    use_recovery_effective = bool(recovery_files) if use_recovery is None else bool(use_recovery)
    collect_slices = use_recovery_effective
    slice_bytes_by_block: dict[int, bytes] = {}
    seen_blocks: set[int] = set()

    h = w = c = None
    block_count = None
    block_size = None
    pad_h = pad_w = None
    quality = None
    coarse_up_arr = None
    coeff_vec = None
    coeff_mask = None
    perm = None
    blocks_h = blocks_w = None
    zigzag = None
    quant = None
    T = None

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

        if version != VERSION_IMG_OLO:
            continue
        off = IMG_HEADER_SIZE
        if off + coarse_len + resid_len > len(data):
            continue
        if B_i <= 0 or block_id < 0 or block_id >= B_i:
            continue

        coarse_payload = data[off: off + coarse_len]
        off += coarse_len
        resid_comp = data[off: off + resid_len]

        if coeff_vec is None:
            if len(coarse_payload) < OLOI_META.size:
                continue
            try:
                magic, meta_ver, block_size_i, quality_i, flags, pad_h_i, pad_w_i, payload_len = OLOI_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOI_MAGIC:
                continue
            if block_size_i <= 0:
                continue

            if meta_ver == 1:
                name = "downsample"
                header_size = OLOI_META.size
                if payload_len > len(coarse_payload) - header_size:
                    continue
                payload_bytes = coarse_payload[header_size: header_size + payload_len]
            elif meta_ver == 2:
                if len(coarse_payload) < OLOI_META_V2.size:
                    continue
                try:
                    (magic, meta_ver, block_size_i, quality_i, flags,
                     pad_h_i, pad_w_i, payload_len, name_len) = OLOI_META_V2.unpack_from(coarse_payload, 0)
                except struct.error:
                    continue
                if magic != OLOI_MAGIC:
                    continue
                name_len = int(name_len)
                header_size = OLOI_META_V2.size
                if name_len < 0 or name_len > len(coarse_payload) - header_size:
                    continue
                name_bytes = coarse_payload[header_size: header_size + name_len]
                name = name_bytes.decode("ascii", errors="ignore").strip().lower() or "downsample"
                payload_start = header_size + name_len
                if payload_len > len(coarse_payload) - payload_start:
                    continue
                payload_bytes = coarse_payload[payload_start: payload_start + payload_len]
            else:
                continue

            model = get_coarse_model(name, kind="image")
            try:
                coarse_up = model.decode(
                    payload_bytes,
                    target_shape=(int(h_i), int(w_i), int(c_i)),
                    block_size=int(block_size_i),
                    quality=int(quality_i),
                )
            except Exception:
                continue

            h, w, c = int(h_i), int(w_i), int(c_i)
            block_count = int(B_i)
            block_size = int(block_size_i)
            pad_h = max(0, int(pad_h_i))
            pad_w = max(0, int(pad_w_i))
            quality = int(np.clip(quality_i, 1, 100))

            coarse_up_arr = np.asarray(coarse_up, dtype=np.int16, order="C")
            blocks_h = max(1, int(math.ceil((h + pad_h) / float(block_size))))
            blocks_w = max(1, int(math.ceil((w + pad_w) / float(block_size))))
            total_coeff = blocks_h * blocks_w * int(c) * (block_size * block_size)
            coeff_vec = np.zeros(total_coeff, dtype=np.int16)
            if return_mask:
                coeff_mask = np.zeros(total_coeff, dtype=bool)

            if block_count > 1:
                perm = _golden_permutation(total_coeff)
            zigzag = _zigzag_indices(block_size)
            quant = _scaled_quant_table(block_size, quality)
            T = _get_dct_matrix(block_size)
        else:
            if (h_i, w_i, c_i, B_i) != (h, w, c, block_count):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if coeff_vec is None:
            continue

        seen_blocks.add(int(block_id))
        if collect_slices and int(block_id) not in slice_bytes_by_block:
            slice_bytes_by_block[int(block_id)] = vals_bytes

        _apply_residual_slice(
            coeff_vec,
            int(block_id),
            int(block_count),
            vals,
            perm,
            coeff_mask if return_mask else None,
        )

    if use_recovery_effective and recovery_files and coeff_vec is not None and block_count is not None:
        missing = [bid for bid in range(int(block_count)) if bid not in seen_blocks]
        if missing:
            lengths, max_len = _residual_slice_lengths(int(coeff_vec.size), int(block_count), 2)
            if max_len > 0:
                rec_chunks = []
                for path in recovery_files:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    chunk = parse_recovery_chunk(data)
                    if chunk is None:
                        continue
                    if chunk.base_kind != REC_KIND_IMAGE or chunk.base_codec_version != VERSION_IMG_OLO:
                        continue
                    if chunk.block_count != int(block_count):
                        continue
                    rec_chunks.append(chunk)

                recovered = recover_missing_slices(
                    block_count=int(block_count),
                    missing_ids=missing,
                    known_slices=slice_bytes_by_block,
                    recovery_chunks=rec_chunks,
                    slice_len=int(max_len),
                )
                if recovered:
                    for block_id, payload in recovered.items():
                        exp_len = lengths[int(block_id)] if int(block_id) < len(lengths) else 0
                        if exp_len <= 0:
                            continue
                        vals = np.frombuffer(payload[:exp_len], dtype="<i2").astype(np.int16, copy=False)
                        _apply_residual_slice(
                            coeff_vec,
                            int(block_id),
                            int(block_count),
                            vals,
                            perm,
                            coeff_mask if return_mask else None,
                        )
                        seen_blocks.add(int(block_id))

    if coeff_vec is None or coarse_up_arr is None or block_size is None or zigzag is None or quant is None or T is None:
        raise ValueError(f"No decodable image chunks found in {in_dir}")

    zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]
    padded_h = blocks_h * block_size
    padded_w = blocks_w * block_size
    residual_full = np.zeros((padded_h, padded_w, int(c)), dtype=np.float32)

    coeff_per_block = block_size * block_size
    blocks_per_channel = blocks_h * blocks_w
    total_expected = coeff_per_block * blocks_per_channel * int(c)
    coeff_use = coeff_vec[:total_expected]
    coeff_use = coeff_use.reshape(int(c), blocks_per_channel, coeff_per_block)

    for ch_idx in range(int(c)):
        ch_blocks = coeff_use[ch_idx]
        for bi in range(blocks_per_channel):
            by = bi // blocks_w
            bx = bi - by * blocks_w
            block_flat = ch_blocks[bi]
            coeff_block = np.zeros((block_size, block_size), dtype=np.float32)
            coeff_block[zz_r, zz_c] = block_flat.astype(np.float32)
            coeff_block *= quant
            spatial = T.T @ coeff_block @ T
            y0 = by * block_size
            x0 = bx * block_size
            residual_full[y0: y0 + block_size, x0: x0 + block_size, ch_idx] = spatial

    residual_crop = residual_full[:h, :w, :]
    recon = coarse_up_arr + residual_crop
    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)

    if return_mask and coeff_mask is not None:
        coeff_per_block = block_size * block_size
        blocks_per_channel = blocks_h * blocks_w
        total_expected = coeff_per_block * blocks_per_channel * int(c)
        coeff_mask_use = coeff_mask[:total_expected].reshape(int(c), blocks_per_channel, coeff_per_block)
        weights = 1.0 / np.maximum(quant, 1e-6)
        weight_flat = weights[zz_r, zz_c].reshape(-1).astype(np.float32)
        weight_sum = float(np.sum(weight_flat)) if weight_flat.size > 0 else 1.0
        conf_blocks = np.zeros(blocks_per_channel, dtype=np.float32)
        for ch_idx in range(int(c)):
            mask_blocks = coeff_mask_use[ch_idx].astype(np.float32)
            conf_blocks += np.sum(mask_blocks * weight_flat[None, :], axis=1) / weight_sum
        conf_blocks /= max(1.0, float(c))
        conf_map = conf_blocks.reshape(int(blocks_h), int(blocks_w))
        return recon_u8, conf_map

    return recon_u8


# ===================== Olonomic v3 field-state helpers =====================

@dataclass
class _ImageFieldStateV3:
    """Internal container for olonomic (version 3) image field state."""

    h: int
    w: int
    c: int
    block_count: int
    block_size: int
    pad_h: int
    pad_w: int
    quality: int
    blocks_h: int
    blocks_w: int
    coarse_payload: bytes
    coarse_up_arr: np.ndarray
    coeff_vec: np.ndarray
    coeff_mask: Optional[np.ndarray] = None
    model_name: str = "downsample"


def _load_image_field_v3(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = True,
) -> _ImageFieldStateV3:
    """Load an olonomic (v3) image field as coefficient-domain state."""
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    recovery_files = []
    if use_recovery is not False:
        recovery_files = sorted(glob.glob(os.path.join(in_dir, "recovery_*.holo")))
    use_recovery_effective = bool(recovery_files) if use_recovery is None else bool(use_recovery)
    collect_slices = use_recovery_effective
    slice_bytes_by_block: dict[int, bytes] = {}
    seen_blocks: set[int] = set()

    h = w = c = None
    block_count = None
    block_size = None
    pad_h = pad_w = None
    quality = None
    coarse_up_arr = None
    coarse_payload_keep: Optional[bytes] = None
    coeff_vec = None
    coeff_mask = None
    perm = None
    blocks_h = blocks_w = None
    model_name = "downsample"

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

        if version != VERSION_IMG_OLO:
            continue
        off = IMG_HEADER_SIZE
        if off + coarse_len + resid_len > len(data):
            continue
        if B_i <= 0 or block_id < 0 or block_id >= B_i:
            continue

        coarse_payload = data[off: off + coarse_len]
        off += coarse_len
        resid_comp = data[off: off + resid_len]

        if coeff_vec is None:
            if len(coarse_payload) < OLOI_META.size:
                continue
            try:
                magic, meta_ver, block_size_i, quality_i, flags, pad_h_i, pad_w_i, payload_len = OLOI_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOI_MAGIC:
                continue
            if block_size_i <= 0:
                continue

            if meta_ver == 1:
                model_name = "downsample"
                header_size = OLOI_META.size
                if payload_len > len(coarse_payload) - header_size:
                    continue
                payload_bytes = coarse_payload[header_size: header_size + payload_len]
            elif meta_ver == 2:
                if len(coarse_payload) < OLOI_META_V2.size:
                    continue
                try:
                    (magic, meta_ver, block_size_i, quality_i, flags,
                     pad_h_i, pad_w_i, payload_len, name_len) = OLOI_META_V2.unpack_from(coarse_payload, 0)
                except struct.error:
                    continue
                if magic != OLOI_MAGIC:
                    continue
                name_len = int(name_len)
                header_size = OLOI_META_V2.size
                if name_len < 0 or name_len > len(coarse_payload) - header_size:
                    continue
                name_bytes = coarse_payload[header_size: header_size + name_len]
                model_name = name_bytes.decode("ascii", errors="ignore").strip().lower() or "downsample"
                payload_start = header_size + name_len
                if payload_len > len(coarse_payload) - payload_start:
                    continue
                payload_bytes = coarse_payload[payload_start: payload_start + payload_len]
            else:
                continue

            model = get_coarse_model(model_name, kind="image")
            try:
                coarse_up = model.decode(
                    payload_bytes,
                    target_shape=(int(h_i), int(w_i), int(c_i)),
                    block_size=int(block_size_i),
                    quality=int(quality_i),
                )
            except Exception:
                continue

            h, w, c = int(h_i), int(w_i), int(c_i)
            block_count = int(B_i)
            block_size = int(block_size_i)
            pad_h = max(0, int(pad_h_i))
            pad_w = max(0, int(pad_w_i))
            quality = int(np.clip(quality_i, 1, 100))

            coarse_up_arr = np.asarray(coarse_up, dtype=np.int16, order="C")
            coarse_payload_keep = bytes(coarse_payload)
            blocks_h = max(1, int(math.ceil((h + pad_h) / float(block_size))))
            blocks_w = max(1, int(math.ceil((w + pad_w) / float(block_size))))
            total_coeff = blocks_h * blocks_w * int(c) * (block_size * block_size)
            coeff_vec = np.zeros(total_coeff, dtype=np.int16)
            if return_mask:
                coeff_mask = np.zeros(total_coeff, dtype=bool)

            if block_count > 1:
                perm = _golden_permutation(total_coeff)
        else:
            if (h_i, w_i, c_i, B_i) != (h, w, c, block_count):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if coeff_vec is None:
            continue

        seen_blocks.add(int(block_id))
        if collect_slices and int(block_id) not in slice_bytes_by_block:
            slice_bytes_by_block[int(block_id)] = vals_bytes

        _apply_residual_slice(
            coeff_vec,
            int(block_id),
            int(block_count),
            vals,
            perm,
            coeff_mask if return_mask else None,
        )

    if use_recovery_effective and recovery_files and coeff_vec is not None and block_count is not None:
        missing = [bid for bid in range(int(block_count)) if bid not in seen_blocks]
        if missing:
            lengths, max_len = _residual_slice_lengths(int(coeff_vec.size), int(block_count), 2)
            if max_len > 0:
                rec_chunks = []
                for path in recovery_files:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    chunk = parse_recovery_chunk(data)
                    if chunk is None:
                        continue
                    if chunk.base_kind != REC_KIND_IMAGE or chunk.base_codec_version != VERSION_IMG_OLO:
                        continue
                    if chunk.block_count != int(block_count):
                        continue
                    rec_chunks.append(chunk)

                recovered = recover_missing_slices(
                    block_count=int(block_count),
                    missing_ids=missing,
                    known_slices=slice_bytes_by_block,
                    recovery_chunks=rec_chunks,
                    slice_len=int(max_len),
                )
                if recovered:
                    for block_id, payload in recovered.items():
                        exp_len = lengths[int(block_id)] if int(block_id) < len(lengths) else 0
                        if exp_len <= 0:
                            continue
                        vals = np.frombuffer(payload[:exp_len], dtype="<i2").astype(np.int16, copy=False)
                        _apply_residual_slice(
                            coeff_vec,
                            int(block_id),
                            int(block_count),
                            vals,
                            perm,
                            coeff_mask if return_mask else None,
                        )
                        seen_blocks.add(int(block_id))

    if coeff_vec is None or coarse_up_arr is None or coarse_payload_keep is None or block_size is None or blocks_h is None or blocks_w is None or quality is None:
        raise ValueError(f"No decodable image v3 chunks found in {in_dir}")

    return _ImageFieldStateV3(
        h=int(h),
        w=int(w),
        c=int(c),
        block_count=int(block_count),
        block_size=int(block_size),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        quality=int(quality),
        blocks_h=int(blocks_h),
        blocks_w=int(blocks_w),
        coarse_payload=coarse_payload_keep,
        coarse_up_arr=coarse_up_arr,
        coeff_vec=coeff_vec,
        coeff_mask=coeff_mask if return_mask else None,
        model_name=str(model_name),
    )


def _confidence_map_image_v3(state: _ImageFieldStateV3) -> np.ndarray:
    """Compute a block-level confidence map from a v3 image coefficient mask."""
    if state.coeff_mask is None:
        return np.ones((int(state.blocks_h), int(state.blocks_w)), dtype=np.float32)

    block_size = int(state.block_size)
    blocks_h = int(state.blocks_h)
    blocks_w = int(state.blocks_w)
    blocks_per_channel = blocks_h * blocks_w
    coeff_per_block = block_size * block_size

    total_expected = coeff_per_block * blocks_per_channel * int(state.c)
    mask_use = state.coeff_mask[:total_expected].reshape(int(state.c), blocks_per_channel, coeff_per_block)

    zigzag = _zigzag_indices(block_size)
    zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]
    quant = _scaled_quant_table(block_size, int(state.quality))

    weights = 1.0 / np.maximum(quant, 1e-6)
    weight_flat = weights[zz_r, zz_c].reshape(-1).astype(np.float32)
    weight_sum = float(np.sum(weight_flat)) if weight_flat.size > 0 else 1.0

    conf_blocks = np.zeros(blocks_per_channel, dtype=np.float32)
    for ch_idx in range(int(state.c)):
        conf_blocks += np.sum(mask_use[ch_idx].astype(np.float32) * weight_flat[None, :], axis=1) / weight_sum
    conf_blocks /= max(1.0, float(state.c))
    return conf_blocks.reshape(blocks_h, blocks_w)


def _render_image_field_v3(
    state: _ImageFieldStateV3,
    *,
    coeff_vec: Optional[np.ndarray] = None,
    coarse_up_arr: Optional[np.ndarray] = None,
    return_mask: bool = False,
) -> np.ndarray:
    """Render an RGB uint8 image from a v3 coefficient-domain field state."""
    coeff_src = state.coeff_vec if coeff_vec is None else np.asarray(coeff_vec)
    coarse_src = state.coarse_up_arr if coarse_up_arr is None else np.asarray(coarse_up_arr)

    h, w, c = int(state.h), int(state.w), int(state.c)
    block_size = int(state.block_size)
    blocks_h = int(state.blocks_h)
    blocks_w = int(state.blocks_w)
    quality = int(state.quality)

    zigzag = _zigzag_indices(block_size)
    zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]
    quant = _scaled_quant_table(block_size, quality)
    T = _get_dct_matrix(block_size)

    padded_h = blocks_h * block_size
    padded_w = blocks_w * block_size
    residual_full = np.zeros((padded_h, padded_w, c), dtype=np.float32)

    coeff_per_block = block_size * block_size
    blocks_per_channel = blocks_h * blocks_w
    total_expected = coeff_per_block * blocks_per_channel * c
    coeff_use = np.asarray(coeff_src[:total_expected], dtype=np.float32, order="C")
    coeff_use = coeff_use.reshape(c, blocks_per_channel, coeff_per_block)

    for ch_idx in range(c):
        ch_blocks = coeff_use[ch_idx]
        for bi in range(blocks_per_channel):
            by = bi // blocks_w
            bx = bi - by * blocks_w
            block_flat = ch_blocks[bi]
            coeff_block = np.zeros((block_size, block_size), dtype=np.float32)
            coeff_block[zz_r, zz_c] = block_flat
            coeff_block *= quant
            spatial = T.T @ coeff_block @ T
            y0 = by * block_size
            x0 = bx * block_size
            residual_full[y0: y0 + block_size, x0: x0 + block_size, ch_idx] = spatial

    residual_crop = residual_full[:h, :w, :]
    recon = coarse_src.astype(np.float32) + residual_crop
    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)

    if return_mask:
        return recon_u8, _confidence_map_image_v3(state)
    return recon_u8


def _write_image_field_v3(
    state: _ImageFieldStateV3,
    out_dir: str,
    *,
    target_chunk_kb: Optional[int] = None,
    max_chunk_bytes: Optional[int] = None,
    block_count: Optional[int] = None,
) -> None:
    """Write a new v3 `.holo` directory from an existing v3 field state."""
    coeff = np.asarray(state.coeff_vec)
    if coeff.dtype != np.int16:
        coeff = np.rint(coeff).astype(np.int64)
        coeff = np.clip(coeff, -32768, 32767).astype(np.int16)

    residual_bytes_total = int(coeff.size) * 2
    coarse_payload = bytes(state.coarse_payload)

    B = int(block_count) if block_count is not None else int(state.block_count)
    if max_chunk_bytes is not None:
        B = _select_block_count_bytes(residual_bytes_total, len(coarse_payload), int(max_chunk_bytes))
    elif target_chunk_kb is not None:
        B = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    B = max(1, min(int(B), max(1, coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(coeff.size) if B > 1 else None
    entries: list[dict] = []

    for block_id in range(B):
        if perm is None or B == 1:
            vals = coeff[block_id::B]
        else:
            idx = perm[block_id::B]
            vals = coeff[idx]

        vals_bytes = vals.astype("<i2", copy=False).tobytes()
        resid_comp = zlib.compress(vals_bytes, level=9)

        header = bytearray()
        header += MAGIC_IMG
        header += struct.pack("B", int(VERSION_IMG_OLO))
        header += struct.pack(">I", int(state.h))
        header += struct.pack(">I", int(state.w))
        header += struct.pack("B", int(state.c))
        header += struct.pack(">I", int(B))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(len(coarse_payload)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{block_id:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_payload + resid_comp)

        score = float(np.sum(vals.astype(np.float64) ** 2))
        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{score:.6f}")
        except OSError:
            pass
        entries.append({"file": os.path.basename(chunk_path), "block_id": int(block_id), "score": float(score)})

    if entries:
        entries.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        _write_chunk_manifest(
            out_dir,
            base_kind="image",
            codec_version=int(VERSION_IMG_OLO),
            block_count=int(B),
            entries=entries,
        )


# ===================== Gauge alignment helpers (v3 image stacking) =====================


def _shift_slices_1d(n: int, shift: int) -> tuple[int, int, int, int]:
    """
    Compute (src0, src1, dst0, dst1) slices for a zero-fill shift.

    We implement translation as:

        out[i] = inp[i - shift]

    Positive `shift` moves content toward higher indices.
    """
    if shift == 0:
        return 0, n, 0, n

    if shift > 0:
        src0 = 0
        src1 = max(0, n - shift)
        dst0 = min(n, shift)
        dst1 = dst0 + (src1 - src0)
        return src0, src1, dst0, dst1

    s = min(n, -shift)
    src0 = s
    src1 = n
    dst0 = 0
    dst1 = max(0, n - s)
    return src0, src1, dst0, dst1


def _shift_array_xy(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """
    Shift a 2D or 3D array by (dy, dx) with zero fill.
    """
    a = np.asarray(arr)
    if a.ndim < 2:
        return a.copy()

    h, w = int(a.shape[0]), int(a.shape[1])
    sy0, sy1, dy0, dy1 = _shift_slices_1d(h, int(dy))
    sx0, sx1, dx0, dx1 = _shift_slices_1d(w, int(dx))

    out = np.zeros_like(a)
    if sy1 <= sy0 or sx1 <= sx0 or dy1 <= dy0 or dx1 <= dx0:
        return out

    out[dy0:dy1, dx0:dx1, ...] = a[sy0:sy1, sx0:sx1, ...]
    return out


def _blockmap_from_coarse_v3(state: _ImageFieldStateV3) -> np.ndarray:
    """
    Build a low-resolution block-grid map from a v3 coarse image.
    """
    block = int(state.block_size)
    h, w, c = int(state.h), int(state.w), int(state.c)
    bh, bw = int(state.blocks_h), int(state.blocks_w)

    coarse = np.asarray(state.coarse_up_arr, dtype=np.float32)
    if coarse.ndim == 3 and c > 1:
        gray = np.mean(coarse, axis=2)
    elif coarse.ndim == 2:
        gray = coarse
    else:
        gray = coarse[..., 0]

    pad_h = max(0, bh * block - h)
    pad_w = max(0, bw * block - w)
    if pad_h > 0 or pad_w > 0:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")

    pooled = gray.reshape(bh, block, bw, block).mean(axis=(1, 3)).astype(np.float32)
    pooled -= float(np.mean(pooled))
    if bh > 1 and bw > 1:
        wy = np.hanning(bh).astype(np.float32)
        wx = np.hanning(bw).astype(np.float32)
        pooled *= (wy[:, None] * wx[None, :])
    return pooled


def _phase_corr_shift2d(ref_map: np.ndarray, tgt_map: np.ndarray) -> tuple[int, int, float]:
    """
    Estimate integer shift to apply to tgt_map so it best aligns to ref_map.
    """
    a = np.asarray(ref_map, dtype=np.float32)
    b = np.asarray(tgt_map, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("phase correlation requires equal shapes")

    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft2(R).real

    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = corr.shape
    dy = int(y) if int(y) <= h // 2 else int(y) - h
    dx = int(x) if int(x) <= w // 2 else int(x) - w
    peak = float(corr[int(y), int(x)])
    return dy, dx, peak


def _estimate_block_shift_image_v3(
    ref: _ImageFieldStateV3,
    tgt: _ImageFieldStateV3,
    *,
    min_peak: float = 0.15,
    max_shift_blocks: Optional[int] = None,
) -> tuple[int, int, float]:
    """
    Estimate a block-grid translation gauge between two v3 image fields.
    """
    ref_map = _blockmap_from_coarse_v3(ref)
    tgt_map = _blockmap_from_coarse_v3(tgt)
    dy, dx, peak = _phase_corr_shift2d(ref_map, tgt_map)

    if not np.isfinite(peak) or peak < float(min_peak):
        return 0, 0, float(peak)

    bh, bw = int(ref.blocks_h), int(ref.blocks_w)
    lim_y = bh // 2
    lim_x = bw // 2
    if max_shift_blocks is not None:
        lim = max(0, int(max_shift_blocks))
        lim_y = min(lim_y, lim)
        lim_x = min(lim_x, lim)

    dy = int(np.clip(dy, -lim_y, lim_y))
    dx = int(np.clip(dx, -lim_x, lim_x))
    return dy, dx, float(peak)


def _shift_image_coeff_blocks_v3(
    state: _ImageFieldStateV3,
    dy_blocks: int,
    dx_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift a v3 coefficient tensor by whole blocks.

    Returns (coeff_shifted, mask_shifted). The returned mask is always provided,
    even if the input state had no coeff_mask.
    """
    c = int(state.c)
    bh = int(state.blocks_h)
    bw = int(state.blocks_w)
    block = int(state.block_size)
    coeff_per_block = block * block
    blocks_per_channel = bh * bw
    total_expected = c * blocks_per_channel * coeff_per_block

    coeff_src = np.asarray(state.coeff_vec[:total_expected], dtype=np.int16, order="C")
    coeff_src = coeff_src.reshape(c, bh, bw, coeff_per_block)

    sy0, sy1, dy0, dy1 = _shift_slices_1d(bh, int(dy_blocks))
    sx0, sx1, dx0, dx1 = _shift_slices_1d(bw, int(dx_blocks))

    coeff_out = np.zeros_like(coeff_src)
    if sy1 > sy0 and sx1 > sx0 and dy1 > dy0 and dx1 > dx0:
        coeff_out[:, dy0:dy1, dx0:dx1, :] = coeff_src[:, sy0:sy1, sx0:sx1, :]

    if state.coeff_mask is not None:
        mask_src = np.asarray(state.coeff_mask[:total_expected], dtype=bool, order="C")
        mask_src = mask_src.reshape(c, bh, bw, coeff_per_block)
        mask_out = np.zeros_like(mask_src)
        if sy1 > sy0 and sx1 > sx0 and dy1 > dy0 and dx1 > dx0:
            mask_out[:, dy0:dy1, dx0:dx1, :] = mask_src[:, sy0:sy1, sx0:sx1, :]
        mask_flat = mask_out.reshape(-1)
    else:
        block_mask = np.zeros((bh, bw), dtype=bool)
        if dy1 > dy0 and dx1 > dx0:
            block_mask[dy0:dy1, dx0:dx1] = True
        per_block = np.repeat(block_mask.reshape(-1), coeff_per_block)
        mask_flat = np.tile(per_block, c).astype(bool)

    coeff_flat = coeff_out.reshape(-1)
    coeff_full = np.zeros_like(state.coeff_vec, dtype=np.int16)
    mask_full = np.zeros_like(state.coeff_vec, dtype=bool)
    coeff_full[:total_expected] = coeff_flat
    mask_full[:total_expected] = mask_flat
    return coeff_full, mask_full


def _decode_image_holo_core(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = False,
) -> np.ndarray:
    chunk_files = _select_chunk_files(in_dir, max_chunks=None, prefer_gain=False)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    for path in chunk_files:
        try:
            with open(path, "rb") as f:
                data = f.read(IMG_HEADER_SIZE)
        except OSError:
            continue
        try:
            version, _, _, _, _, _, _, _ = _parse_image_header(data)
        except ChunkFormatError:
            continue
        if version == VERSION_IMG_OLO:
            return _decode_image_holo_core_v3(
                in_dir,
                max_chunks=max_chunks,
                prefer_gain=prefer_gain,
                use_recovery=use_recovery,
                return_mask=return_mask,
            )
        return _decode_image_holo_core_v1_v2(
            in_dir,
            max_chunks=max_chunks,
            prefer_gain=prefer_gain,
            return_mask=return_mask,
        )

    raise ValueError(f"No decodable image chunks found in {in_dir}")


def decode_image_holo_dir(
    in_dir: str,
    output_path: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
) -> None:
    """
    Decode an image from a holographic directory and save it to disk.

    Parameters
    ----------
    in_dir : str
        Directory containing chunk_*.holo files.
    output_path : str
        Path where the decoded image will be written.
    max_chunks : int, optional
        If given, only the first max_chunks chunks are used. This lets you
        simulate partial coverage, for example to measure graceful degradation.

    Example
    -------
    Decode using all chunks:

        decode_image_holo_dir("frame.png.holo", "frame_recon.png")

    Decode using only 8 chunks (out of many):

        decode_image_holo_dir("frame.png.holo", "frame_recon_8.png",
                              max_chunks=8)
    """
    recon = _decode_image_holo_core(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
    )
    save_image_rgb_u8(recon, output_path)


def decode_image_holo_dir_meta(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode an image and return (reconstruction, confidence_map).
    """
    recon, conf = _decode_image_holo_core(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
        return_mask=True,
    )
    return recon, conf


def decode_image_olonomic_holo_dir(
    in_dir: str,
    output_path: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
) -> None:
    """
    Decode an olonomic (version 3) holographic image directory.
    """
    recon = _decode_image_holo_core_v3(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
    )
    save_image_rgb_u8(recon, output_path)


def stack_image_holo_dirs(
    in_dirs: Sequence[str],
    output_path: str,
    *,
    max_chunks: Optional[int] = None,
    gauge_align: bool = True,
    min_gauge_peak: float = 0.15,
    max_shift_blocks: Optional[int] = None,
) -> None:
    """
    Stack multiple holographic image directories to improve SNR over time.

    For olonomic v3 fields, if all decodable inputs are v3, stacking happens
    in coefficient space (genotypic stacking). Otherwise this falls back to a
    pixel-domain average (phenotypic stacking).

    When `gauge_align` is enabled, we estimate an integer block-grid translation
    between exposures (via phase correlation of the coarse base) and shift the
    coefficient tensors before summing.
    """
    v3_states: list[_ImageFieldStateV3] = []
    all_v3 = True

    for d in in_dirs:
        if not os.path.isdir(d):
            continue
        chunk_files = sorted(glob.glob(os.path.join(d, "chunk_*.holo")))
        if not chunk_files:
            continue
        try:
            with open(chunk_files[0], "rb") as f:
                head = f.read(IMG_HEADER_SIZE)
            version, *_ = _parse_image_header(head)
        except Exception:
            all_v3 = False
            break
        if version != VERSION_IMG_OLO:
            all_v3 = False
            break
        try:
            st = _load_image_field_v3(
                d,
                max_chunks=max_chunks,
                prefer_gain=False,
                use_recovery=None,
                return_mask=True,
            )
            v3_states.append(st)
        except Exception:
            all_v3 = False
            break

    if all_v3 and v3_states:
        ref = v3_states[0]
        total_coeff = int(ref.coeff_vec.size)

        coeff_sum = np.zeros(total_coeff, dtype=np.float64)
        coeff_w = np.zeros(total_coeff, dtype=np.float64)
        coarse_sum = np.zeros((int(ref.h), int(ref.w), int(ref.c)), dtype=np.float64)
        coarse_w = np.zeros((int(ref.h), int(ref.w)), dtype=np.float64)

        for st in v3_states:
            if (st.h, st.w, st.c, st.block_size, st.blocks_h, st.blocks_w) != (
                ref.h, ref.w, ref.c, ref.block_size, ref.blocks_h, ref.blocks_w
            ):
                continue
            if int(st.coeff_vec.size) != total_coeff:
                continue

            dy_b = 0
            dx_b = 0
            if gauge_align and st is not ref:
                try:
                    dy_b, dx_b, _peak = _estimate_block_shift_image_v3(
                        ref,
                        st,
                        min_peak=float(min_gauge_peak),
                        max_shift_blocks=max_shift_blocks,
                    )
                except Exception:
                    dy_b, dx_b = 0, 0

            dy_px = int(dy_b) * int(ref.block_size)
            dx_px = int(dx_b) * int(ref.block_size)

            coeff_shift, mask_shift = _shift_image_coeff_blocks_v3(st, int(dy_b), int(dx_b))
            coarse_shift = _shift_array_xy(np.asarray(st.coarse_up_arr, dtype=np.float32), dy_px, dx_px)
            valid = _shift_array_xy(np.ones((int(ref.h), int(ref.w)), dtype=np.float32), dy_px, dx_px)

            coarse_sum += coarse_shift.astype(np.float64) * valid[..., None].astype(np.float64)
            coarse_w += valid.astype(np.float64)

            m = np.asarray(mask_shift, dtype=bool)
            coeff_sum[m] += coeff_shift[m].astype(np.float64)
            coeff_w[m] += 1.0

        if float(np.max(coarse_w)) <= 0.0:
            raise ValueError("No compatible v3 images found to stack")

        coarse_avg = (coarse_sum / np.maximum(coarse_w[..., None], 1.0)).astype(np.float32)
        coeff_avg = np.zeros(total_coeff, dtype=np.float32)
        np.divide(coeff_sum, np.maximum(coeff_w, 1.0), out=coeff_avg, where=coeff_w > 0.0)

        stacked = _render_image_field_v3(ref, coeff_vec=coeff_avg, coarse_up_arr=coarse_avg)
        save_image_rgb_u8(stacked, output_path)
        return

    acc = None
    count = 0

    for d in in_dirs:
        if not os.path.isdir(d):
            continue
        try:
            img = _decode_image_holo_core(d, max_chunks=max_chunks)
        except Exception:
            continue

        img_f = img.astype(np.float32)
        if acc is None:
            acc = img_f
        else:
            if acc.shape != img_f.shape:
                continue
            acc += img_f
        count += 1

    if acc is None or count == 0:
        raise ValueError("No decodable and compatible images found in the given .holo directories")

    stacked = np.clip(acc / float(count), 0.0, 255.0).astype(np.uint8)
    save_image_rgb_u8(stacked, output_path)


# ===================== Audio helpers and codec (WAV) =====================

def _read_wav_as_int16(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Read a PCM WAV file as int16 samples.

    Supports 16-bit PCM (kept as-is) and 24-bit PCM (down-converted to int16).
    Returns samples with shape (n_frames, n_channels).

    Parameters
    ----------
    path : str
        Path to the WAV file.

    Returns
    -------
    samples : np.ndarray
        Int16 samples, shape (n_frames, n_channels).
    sample_rate : int
        Sample rate in Hz.
    channels : int
        Number of channels.
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
        # Manual unpack of 24-bit PCM into int32 with sign extension, then downscale to int16
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size != total * 3:
            raise ValueError("Invalid 24-bit WAV payload size")
        b = b.reshape(-1, 3)
        vals = (b[:, 0].astype(np.int32) |
                (b[:, 1].astype(np.int32) << 8) |
                (b[:, 2].astype(np.int32) << 16))
        mask = 1 << 23
        vals = (vals ^ mask) - mask  # sign extend 24-bit
        data = (vals >> 8).astype(np.int16)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    return data.reshape(-1, ch), int(sr), int(ch)


def _write_wav_int16(path: str, audio: np.ndarray, sr: int) -> None:
    """
    Write an int16 PCM WAV file.

    Parameters
    ----------
    path : str
        Output path.
    audio : np.ndarray
        Int16 samples, shape (n_frames, n_channels).
    sr : int
        Sample rate in Hz.
    """
    audio = np.asarray(audio, dtype=np.int16)
    n_frames, ch = audio.shape
    with wave.open(path, "wb") as wf:
        wf.setnchannels(int(ch))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio.astype("<i2", copy=False).tobytes())


def _stft_frame_count(n_samples: int, n_fft: int, hop: int) -> Tuple[int, int]:
    if n_samples <= 0:
        return 0, 0
    frames = max(1, int(math.ceil((n_samples - n_fft) / float(hop))) + 1)
    padded_len = n_fft + (frames - 1) * hop
    pad = max(0, padded_len - n_samples)
    return frames, pad


def _stft_1d(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> Tuple[np.ndarray, int]:
    frames, pad = _stft_frame_count(x.size, n_fft, hop)
    padded = np.pad(x.astype(np.float32, copy=False), (0, pad), mode="constant")
    spec = np.empty((frames, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(frames):
        start = i * hop
        segment = padded[start:start + n_fft]
        spec[i] = np.fft.rfft(segment * window)
    return spec, pad


def _istft_1d(spec: np.ndarray, n_fft: int, hop: int, window: np.ndarray, out_len: int) -> np.ndarray:
    frames = spec.shape[0]
    padded_len = n_fft + (frames - 1) * hop
    out = np.zeros(padded_len, dtype=np.float64)
    norm = np.zeros(padded_len, dtype=np.float64)
    for i in range(frames):
        start = i * hop
        frame = np.fft.irfft(spec[i], n=n_fft)
        out[start:start + n_fft] += frame * window
        norm[start:start + n_fft] += window * window
    norm = np.maximum(norm, 1e-8)
    out /= norm
    return out[:out_len]


def _audio_quant_steps(n_bins: int, quality: int, n_fft: int) -> np.ndarray:
    q = int(np.clip(quality, 1, 100))
    norm = max(1.0, float(n_fft))
    base = max(1.0, (120.0 - float(q)) / 20.0)
    idx = np.arange(n_bins, dtype=np.float32)
    scale = 1.0 + idx / max(1.0, float(n_bins - 1))
    return (base / norm) * scale


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
    Encode a PCM WAV file into holographic audio chunks.

    The audio path mirrors the image path:
      - A coarse track is built by sampling `coarse_max_frames` points over
        the full duration and linearly interpolating back to n_frames.
      - The residual (audio - coarse_up) is computed and stored in int16.

    Each chunk carries:
      - the complete coarse sequence (int16, zlib-compressed),
      - a different slice of the residual, interleaved via golden permutation
        when version >= 2.

    If residual amplitudes do not fit in int16, they are clipped to
    [-32768, 32767] and this condition is flagged in the header. This trades
    exactness for stability, avoiding wrap-around artifacts.

    Parameters
    ----------
    input_wav : str
        Path to input WAV file (16- or 24-bit PCM).
    out_dir : str
        Target directory for chunks (e.g. "track.wav.holo").
    block_count : int, optional
        Desired number of chunks. Overridden if target_chunk_kb is given.
    coarse_max_frames : int, optional
        Maximum number of coarse samples used to approximate the envelope.
    target_chunk_kb : int, optional
        Approximate chunk size in kilobytes.
    version : int, optional
        Codec version (1 or 2).

    Example
    -------
    Encode audio into ~32 KB chunks:

        encode_audio_holo_dir(
            "track.wav",
            "track.wav.holo",
            target_chunk_kb=32,
        )
    """
    if version not in (1, VERSION_AUD):
        raise ValueError(f"Unsupported audio codec version: {version}")

    audio, sr, ch = _read_wav_as_int16(input_wav)
    n_frames = int(audio.shape[0])
    if n_frames < 2:
        raise ValueError("Audio too short to encode")

    # Build a coarse representation by sub-sampling in time
    coarse_len = int(min(max(2, int(coarse_max_frames)), n_frames))
    idx = np.linspace(0, n_frames - 1, coarse_len, dtype=np.int64)
    coarse = audio[idx]

    # Linearly interpolate coarse back to full length
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


def encode_audio_olonomic_holo_dir(
    input_wav: str,
    out_dir: str,
    *,
    block_count: int = 16,
    coarse_max_frames: int = 2048,
    coarse_model: str = "downsample",
    target_chunk_kb: Optional[int] = None,
    quality: int = 50,
    n_fft: int = 512,
    hop: Optional[int] = None,
    recovery: Optional[str] = None,
    overhead: float = 0.0,
    recovery_seed: Optional[int] = None,
    version: int = VERSION_AUD_OLO,
) -> None:
    if version != VERSION_AUD_OLO:
        raise ValueError(f"Unsupported olonomic audio codec version: {version}")

    audio, sr, ch = _read_wav_as_int16(input_wav)
    n_frames = int(audio.shape[0])
    if n_frames < 2:
        raise ValueError("Audio too short to encode")

    n_fft = int(n_fft)
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    hop_val = int(hop) if hop is not None else n_fft // 2
    hop_val = max(1, hop_val)
    quality_u8 = int(np.clip(quality, 1, 100))

    model = get_coarse_model(coarse_model, kind="audio")
    model_name = model.name
    name_bytes = model_name.encode("ascii", errors="ignore") or b"downsample"
    if len(name_bytes) > 255:
        name_bytes = name_bytes[:255]

    if model_name in ("downsample", "ae_latent"):
        coarse_len = int(min(max(2, int(coarse_max_frames)), n_frames))
    else:
        coarse_len = int(n_frames)

    coarse_payload_bytes = model.encode(
        audio,
        coarse_len=int(coarse_len),
        n_fft=int(n_fft),
        hop=int(hop_val),
        quality=int(quality_u8),
    )
    coarse_up = model.decode(
        coarse_payload_bytes,
        target_shape=(int(n_frames), int(ch)),
        coarse_len=int(coarse_len),
        n_fft=int(n_fft),
        hop=int(hop_val),
        quality=int(quality_u8),
    )
    coarse_up = np.asarray(coarse_up, dtype=np.int16, order="C")

    residual_f = (audio.astype(np.int32) - coarse_up.astype(np.int32)).astype(np.float32)

    window = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    bins = n_fft // 2 + 1
    steps = _audio_quant_steps(bins, quality_u8, n_fft)

    coeff_parts: list[np.ndarray] = []
    clipped = False
    frames_expected = None

    for ch_idx in range(ch):
        spec, _ = _stft_1d(residual_f[:, ch_idx], n_fft, hop_val, window)
        if frames_expected is None:
            frames_expected = spec.shape[0]
        elif spec.shape[0] != frames_expected:
            raise ValueError("STFT frame mismatch across channels")
        spec_scaled = spec / float(n_fft)
        q_re_f = np.rint(np.real(spec_scaled) / steps)
        q_im_f = np.rint(np.imag(spec_scaled) / steps)
        q_re = np.clip(q_re_f, -32768.0, 32767.0)
        q_im = np.clip(q_im_f, -32768.0, 32767.0)
        if not clipped and (np.any(q_re != q_re_f) or np.any(q_im != q_im_f)):
            clipped = True
        q_re_i = q_re.astype(np.int16, copy=False)
        q_im_i = q_im.astype(np.int16, copy=False)
        interleaved = np.empty(q_re_i.size * 2, dtype=np.int16)
        interleaved[0::2] = q_re_i.reshape(-1)
        interleaved[1::2] = q_im_i.reshape(-1)
        coeff_parts.append(interleaved)

    residual_coeff = np.concatenate(coeff_parts, axis=0) if coeff_parts else np.zeros(0, dtype=np.int16)
    residual_bytes_total = residual_coeff.size * 2

    meta_flags = OLOA_FLAG_COEFF_CLIPPED if clipped else 0
    coarse_payload = OLOA_META.pack(
        OLOA_MAGIC,
        2,  # meta version
        quality_u8,
        meta_flags,
        int(len(name_bytes)),
        int(n_fft),
        int(hop_val),
    ) + name_bytes + coarse_payload_bytes

    if target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), max(1, residual_coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(residual_coeff.size) if block_count > 1 else None
    use_recovery = bool(recovery) and str(recovery).lower() == "rlnc" and float(overhead) > 0.0
    slice_bytes = [b""] * block_count if use_recovery else None

    flags = 0
    if clipped:
        flags |= AUD_FLAG_RESID_CLIPPED

    chunks: list[tuple[float, int, bytes]] = []

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_coeff[block_id::block_count]
        else:
            idxb = perm[block_id::block_count]
            vals = residual_coeff[idxb]

        vals_bytes = vals.astype("<i2", copy=False).tobytes()
        if slice_bytes is not None:
            slice_bytes[block_id] = vals_bytes
        resid_comp = zlib.compress(vals_bytes, level=9)
        score = float(np.sum(vals.astype(np.float64) ** 2))
        chunks.append((score, block_id, resid_comp))

    chunks.sort(key=lambda x: x[0], reverse=True)

    entries: list[dict] = []
    for out_idx, (score, block_id, resid_comp) in enumerate(chunks):
        header = bytearray()
        header += MAGIC_AUD
        header += struct.pack("B", int(version))
        header += struct.pack("B", int(ch))
        header += struct.pack("B", 2)
        header += struct.pack("B", int(flags))
        header += struct.pack(">I", int(sr))
        header += struct.pack(">I", int(n_frames))
        header += struct.pack(">I", int(block_count))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(coarse_len))
        header += struct.pack(">I", int(len(coarse_payload)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{out_idx:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_payload + resid_comp)

        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{score:.6f}")
        except OSError:
            pass
        entries.append(
            {"file": os.path.basename(chunk_path), "block_id": int(block_id), "score": float(score)}
        )

    if slice_bytes is not None:
        recovery_chunks = build_recovery_chunks(
            slice_bytes,
            base_kind=REC_KIND_AUDIO,
            base_codec_version=int(version),
            overhead=float(overhead),
            seed=recovery_seed,
        )
        for idx, chunk_bytes in enumerate(recovery_chunks):
            rec_path = os.path.join(out_dir, f"recovery_{idx:04d}.holo")
            with open(rec_path, "wb") as f:
                f.write(chunk_bytes)

    if entries:
        entries.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        _write_chunk_manifest(
            out_dir,
            base_kind="audio",
            codec_version=int(version),
            block_count=int(block_count),
            entries=entries,
        )


def _parse_audio_header(data: bytes) -> Tuple[int, int, int, int, int, int, int, int, int, int, int]:
    """
    Parse the fixed-size audio header from a chunk payload.

    Returns:
      (version, channels, sample_width, flags,
       sample_rate, n_frames, block_count, block_id,
       coarse_len, coarse_size, resid_size)
    """
    if len(data) < AUD_HEADER_SIZE:
        raise ChunkFormatError("Audio chunk too small for header")
    off = 0
    if data[off:off + 4] != MAGIC_AUD:
        raise ChunkFormatError("Bad audio magic")
    off += 4

    version = data[off]; off += 1
    if version not in (1, VERSION_AUD, VERSION_AUD_OLO):
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


def _decode_audio_holo_core_v1_v2(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    return_mask: bool = False,
) -> Tuple[np.ndarray, int]:
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    sr = ch = n_frames = None
    block_count = None
    coarse_len = None
    version_used = None

    coarse_up = None
    residual_flat = None
    mask_flat = None
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

        if version == VERSION_AUD_OLO:
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
            # First valid chunk drives allocation
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
            if return_mask:
                mask_flat = np.zeros(n_frames * ch, dtype=bool)

            if version_used == VERSION_AUD and block_count > 1:
                perm = _golden_permutation(residual_flat.size)
        else:
            # Make sure subsequent chunks are compatible
            if (sr_i, ch_i, n_frames_i, B_i, coarse_len_i, version) != (sr, ch, n_frames, block_count, coarse_len, version_used):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        _apply_residual_slice(
            residual_flat,
            int(block_id),
            int(block_count),
            vals,
            perm,
            mask_flat if return_mask else None,
        )

    if residual_flat is None or coarse_up is None:
        raise ValueError(f"No decodable audio chunks found in {in_dir}")

    residual = residual_flat.reshape(n_frames, ch)
    recon = coarse_up.astype(np.int32) + residual.astype(np.int32)
    recon = np.clip(recon, -32768, 32767).astype(np.int16)

    if return_mask and mask_flat is not None:
        mask_curve = mask_flat.reshape(int(n_frames), int(ch)).mean(axis=1).astype(np.float32)
        return recon, int(sr), mask_curve

    return recon, int(sr)


def _decode_audio_holo_core_v3(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = False,
) -> Tuple[np.ndarray, int]:
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    recovery_files = []
    if use_recovery is not False:
        recovery_files = sorted(glob.glob(os.path.join(in_dir, "recovery_*.holo")))
    use_recovery_effective = bool(recovery_files) if use_recovery is None else bool(use_recovery)
    collect_slices = use_recovery_effective
    slice_bytes_by_block: dict[int, bytes] = {}
    seen_blocks: set[int] = set()

    sr = ch = n_frames = None
    block_count = None
    coarse_len = None
    n_fft = None
    hop = None
    quality = None
    coeff_vec = None
    coeff_mask = None
    perm = None
    frames = None
    bins = None
    coarse_up = None
    used_chunks = 0

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

        if version != VERSION_AUD_OLO:
            continue
        if sampwidth != 2 or B_i <= 0 or block_id < 0 or block_id >= B_i:
            continue

        off = AUD_HEADER_SIZE
        if off + coarse_size + resid_size > len(data):
            continue

        coarse_payload = data[off:off + coarse_size]
        off += coarse_size
        resid_comp = data[off:off + resid_size]

        if coeff_vec is None:
            if len(coarse_payload) < OLOA_META.size:
                continue
            try:
                magic, meta_ver, quality_i, meta_flags, name_len, n_fft_i, hop_i = OLOA_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOA_MAGIC:
                continue
            payload_start = OLOA_META.size
            if meta_ver == 1:
                name = "downsample"
            elif meta_ver == 2:
                name_len = int(name_len)
                if name_len < 0 or name_len > len(coarse_payload) - payload_start:
                    continue
                name_bytes = coarse_payload[payload_start: payload_start + name_len]
                name = name_bytes.decode("ascii", errors="ignore").strip().lower() or "downsample"
                payload_start += name_len
            else:
                continue
            payload_bytes = coarse_payload[payload_start:]

            sr = int(sr_i)
            ch = int(ch_i)
            n_frames = int(n_frames_i)
            block_count = int(B_i)
            coarse_len = int(coarse_len_i)
            n_fft = int(n_fft_i)
            hop = max(1, int(hop_i))
            quality = int(quality_i)
            if n_fft <= 0 or coarse_len <= 0:
                coeff_vec = None
                coarse_up = None
                continue
            model = get_coarse_model(name, kind="audio")
            try:
                coarse_up = model.decode(
                    payload_bytes,
                    target_shape=(int(n_frames), int(ch)),
                    coarse_len=int(coarse_len),
                    n_fft=int(n_fft),
                    hop=int(hop),
                    quality=int(quality),
                )
            except Exception:
                coarse_up = None
                continue
            coarse_up = np.asarray(coarse_up, dtype=np.int16, order="C")

            frames, _ = _stft_frame_count(n_frames, n_fft, hop)
            bins = n_fft // 2 + 1
            total_coeff = int(ch) * frames * bins * 2
            coeff_vec = np.zeros(total_coeff, dtype=np.int16)
            if return_mask:
                coeff_mask = np.zeros(total_coeff, dtype=bool)

            if block_count > 1:
                perm = _golden_permutation(total_coeff)
        else:
            if (sr_i, ch_i, n_frames_i, B_i, coarse_len_i) != (sr, ch, n_frames, block_count, coarse_len):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if coeff_vec is None:
            continue

        seen_blocks.add(int(block_id))
        if collect_slices and int(block_id) not in slice_bytes_by_block:
            slice_bytes_by_block[int(block_id)] = vals_bytes

        _apply_residual_slice(
            coeff_vec,
            int(block_id),
            int(block_count),
            vals,
            perm,
            coeff_mask if return_mask else None,
        )
        used_chunks += 1

    if use_recovery_effective and recovery_files and coeff_vec is not None and block_count is not None:
        missing = [bid for bid in range(int(block_count)) if bid not in seen_blocks]
        if missing:
            lengths, max_len = _residual_slice_lengths(int(coeff_vec.size), int(block_count), 2)
            if max_len > 0:
                rec_chunks = []
                for path in recovery_files:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    chunk = parse_recovery_chunk(data)
                    if chunk is None:
                        continue
                    if chunk.base_kind != REC_KIND_AUDIO or chunk.base_codec_version != VERSION_AUD_OLO:
                        continue
                    if chunk.block_count != int(block_count):
                        continue
                    rec_chunks.append(chunk)

                recovered = recover_missing_slices(
                    block_count=int(block_count),
                    missing_ids=missing,
                    known_slices=slice_bytes_by_block,
                    recovery_chunks=rec_chunks,
                    slice_len=int(max_len),
                )
                if recovered:
                    for block_id, payload in recovered.items():
                        exp_len = lengths[int(block_id)] if int(block_id) < len(lengths) else 0
                        if exp_len <= 0:
                            continue
                        vals = np.frombuffer(payload[:exp_len], dtype="<i2").astype(np.int16, copy=False)
                        _apply_residual_slice(
                            coeff_vec,
                            int(block_id),
                            int(block_count),
                            vals,
                            perm,
                            coeff_mask if return_mask else None,
                        )
                        seen_blocks.add(int(block_id))

    if coeff_vec is None or coarse_up is None or sr is None or ch is None or n_frames is None or n_fft is None or hop is None or quality is None or frames is None or bins is None:
        raise ValueError(f"No decodable audio chunks found in {in_dir}")

    coeff_use = coeff_vec.reshape(int(ch), frames, bins * 2)
    window = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    steps = _audio_quant_steps(bins, quality, n_fft)
    residual = np.zeros((n_frames, int(ch)), dtype=np.float64)

    for ch_idx in range(int(ch)):
        q_re = coeff_use[ch_idx, :, 0::2].astype(np.float32)
        q_im = coeff_use[ch_idx, :, 1::2].astype(np.float32)
        spec = (q_re + 1j * q_im) * steps[None, :]
        spec *= float(n_fft)
        res_ch = _istft_1d(spec, n_fft, hop, window, n_frames)
        residual[:, ch_idx] = res_ch

    coverage = 1.0
    residual *= coverage

    recon = coarse_up.astype(np.int32) + np.round(residual).astype(np.int32)
    recon = np.clip(recon, -32768, 32767).astype(np.int16)
    if return_mask and coeff_mask is not None and frames is not None and bins is not None:
        weights = 1.0 / (1.0 + np.arange(int(bins), dtype=np.float32))
        weights = np.repeat(weights, 2)
        weight_sum = float(np.sum(weights)) if weights.size > 0 else 1.0
        mask_use = coeff_mask.reshape(int(ch), int(frames), int(bins) * 2)
        conf_frames = np.zeros(int(frames), dtype=np.float32)
        for ch_idx in range(int(ch)):
            conf_frames += np.sum(mask_use[ch_idx].astype(np.float32) * weights[None, :], axis=1) / weight_sum
        conf_frames /= max(1.0, float(ch))
        if int(frames) > 1:
            frame_pos = np.linspace(0.0, float(n_frames - 1), int(frames))
            conf_curve = np.interp(np.arange(int(n_frames)), frame_pos, conf_frames).astype(np.float32)
        else:
            conf_curve = np.full(int(n_frames), float(conf_frames[0]), dtype=np.float32)
        return recon, int(sr), conf_curve

    return recon, int(sr)


# ===================== Olonomic v3 audio field-state helpers =====================

@dataclass
class _AudioFieldStateV3:
    """Internal container for olonomic (version 3) audio field state."""

    sr: int
    ch: int
    n_frames: int
    block_count: int
    coarse_len: int
    n_fft: int
    hop: int
    quality: int
    flags: int
    coarse_payload: bytes
    coarse_up: np.ndarray
    coeff_vec: np.ndarray
    coeff_mask: Optional[np.ndarray] = None
    frames: int = 0
    bins: int = 0
    model_name: str = "downsample"


def _load_audio_field_v3(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = True,
) -> _AudioFieldStateV3:
    """Load an olonomic (v3) audio field as coefficient-domain state."""
    chunk_files = _select_chunk_files(in_dir, max_chunks=max_chunks, prefer_gain=prefer_gain)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    recovery_files = []
    if use_recovery is not False:
        recovery_files = sorted(glob.glob(os.path.join(in_dir, "recovery_*.holo")))
    use_recovery_effective = bool(recovery_files) if use_recovery is None else bool(use_recovery)
    collect_slices = use_recovery_effective
    slice_bytes_by_block: dict[int, bytes] = {}
    seen_blocks: set[int] = set()

    sr = ch = n_frames = None
    block_count = None
    coarse_len = None
    n_fft = None
    hop = None
    quality = None
    flags_keep: Optional[int] = None
    coarse_payload_keep: Optional[bytes] = None
    coarse_up = None
    coeff_vec = None
    coeff_mask = None
    perm = None
    frames = None
    bins = None
    model_name = "downsample"

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

        if version != VERSION_AUD_OLO:
            continue
        if sampwidth != 2 or B_i <= 0 or block_id < 0 or block_id >= B_i:
            continue

        off = AUD_HEADER_SIZE
        if off + coarse_size + resid_size > len(data):
            continue

        coarse_payload = data[off:off + coarse_size]
        off += coarse_size
        resid_comp = data[off:off + resid_size]

        if coeff_vec is None:
            if len(coarse_payload) < OLOA_META.size:
                continue
            try:
                magic, meta_ver, quality_i, meta_flags, name_len, n_fft_i, hop_i = OLOA_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOA_MAGIC:
                continue

            payload_start = OLOA_META.size
            if meta_ver == 1:
                model_name = "downsample"
            elif meta_ver == 2:
                name_len = int(name_len)
                if name_len < 0 or name_len > len(coarse_payload) - payload_start:
                    continue
                name_bytes = coarse_payload[payload_start: payload_start + name_len]
                model_name = name_bytes.decode("ascii", errors="ignore").strip().lower() or "downsample"
                payload_start += name_len
            else:
                continue
            payload_bytes = coarse_payload[payload_start:]

            sr = int(sr_i)
            ch = int(ch_i)
            n_frames = int(n_frames_i)
            block_count = int(B_i)
            coarse_len = int(coarse_len_i)
            n_fft = int(n_fft_i)
            hop = max(1, int(hop_i))
            quality = int(quality_i)
            flags_keep = int(flags)

            if n_fft <= 0 or coarse_len <= 0:
                coeff_vec = None
                coarse_up = None
                continue

            model = get_coarse_model(model_name, kind="audio")
            try:
                coarse_up = model.decode(
                    payload_bytes,
                    target_shape=(int(n_frames), int(ch)),
                    coarse_len=int(coarse_len),
                    n_fft=int(n_fft),
                    hop=int(hop),
                    quality=int(quality),
                )
            except Exception:
                coarse_up = None
                continue
            coarse_up = np.asarray(coarse_up, dtype=np.int16, order="C")

            frames, _ = _stft_frame_count(n_frames, n_fft, hop)
            bins = n_fft // 2 + 1
            total_coeff = int(ch) * int(frames) * int(bins) * 2
            coeff_vec = np.zeros(total_coeff, dtype=np.int16)
            if return_mask:
                coeff_mask = np.zeros(total_coeff, dtype=bool)

            coarse_payload_keep = bytes(coarse_payload)

            if block_count > 1:
                perm = _golden_permutation(total_coeff)
        else:
            if (sr_i, ch_i, n_frames_i, B_i, coarse_len_i) != (sr, ch, n_frames, block_count, coarse_len):
                continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if coeff_vec is None:
            continue

        seen_blocks.add(int(block_id))
        if collect_slices and int(block_id) not in slice_bytes_by_block:
            slice_bytes_by_block[int(block_id)] = vals_bytes

        _apply_residual_slice(
            coeff_vec,
            int(block_id),
            int(block_count),
            vals,
            perm,
            coeff_mask if return_mask else None,
        )

    if use_recovery_effective and recovery_files and coeff_vec is not None and block_count is not None:
        missing = [bid for bid in range(int(block_count)) if bid not in seen_blocks]
        if missing:
            lengths, max_len = _residual_slice_lengths(int(coeff_vec.size), int(block_count), 2)
            if max_len > 0:
                rec_chunks = []
                for path in recovery_files:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                    except OSError:
                        continue
                    chunk = parse_recovery_chunk(data)
                    if chunk is None:
                        continue
                    if chunk.base_kind != REC_KIND_AUDIO or chunk.base_codec_version != VERSION_AUD_OLO:
                        continue
                    if chunk.block_count != int(block_count):
                        continue
                    rec_chunks.append(chunk)

                recovered = recover_missing_slices(
                    block_count=int(block_count),
                    missing_ids=missing,
                    known_slices=slice_bytes_by_block,
                    recovery_chunks=rec_chunks,
                    slice_len=int(max_len),
                )
                if recovered:
                    for block_id, payload in recovered.items():
                        exp_len = lengths[int(block_id)] if int(block_id) < len(lengths) else 0
                        if exp_len <= 0:
                            continue
                        vals = np.frombuffer(payload[:exp_len], dtype="<i2").astype(np.int16, copy=False)
                        _apply_residual_slice(
                            coeff_vec,
                            int(block_id),
                            int(block_count),
                            vals,
                            perm,
                            coeff_mask if return_mask else None,
                        )
                        seen_blocks.add(int(block_id))

    if coeff_vec is None or coarse_up is None or sr is None or ch is None or n_frames is None or coarse_len is None or n_fft is None or hop is None or quality is None or frames is None or bins is None or coarse_payload_keep is None or flags_keep is None:
        raise ValueError(f"No decodable audio v3 chunks found in {in_dir}")

    return _AudioFieldStateV3(
        sr=int(sr),
        ch=int(ch),
        n_frames=int(n_frames),
        block_count=int(block_count),
        coarse_len=int(coarse_len),
        n_fft=int(n_fft),
        hop=int(hop),
        quality=int(quality),
        flags=int(flags_keep),
        coarse_payload=coarse_payload_keep,
        coarse_up=coarse_up,
        coeff_vec=coeff_vec,
        coeff_mask=coeff_mask if return_mask else None,
        frames=int(frames),
        bins=int(bins),
        model_name=str(model_name),
    )


def _confidence_frames_audio_v3(state: _AudioFieldStateV3) -> np.ndarray:
    """Return per-STFT-frame confidence values for a v3 audio field."""
    if state.coeff_mask is None:
        return np.ones(int(state.frames), dtype=np.float32)
    frames = int(state.frames)
    bins = int(state.bins)
    ch = int(state.ch)
    weights = 1.0 / (1.0 + np.arange(bins, dtype=np.float32))
    weights = np.repeat(weights, 2)
    weight_sum = float(np.sum(weights)) if weights.size > 0 else 1.0
    mask_use = state.coeff_mask.reshape(ch, frames, bins * 2)
    conf_frames = np.zeros(frames, dtype=np.float32)
    for ch_idx in range(ch):
        conf_frames += np.sum(mask_use[ch_idx].astype(np.float32) * weights[None, :], axis=1) / weight_sum
    conf_frames /= max(1.0, float(ch))
    return np.clip(conf_frames, 0.0, 1.0)


def _confidence_curve_audio_v3(state: _AudioFieldStateV3) -> np.ndarray:
    """Return per-sample confidence curve for a v3 audio field."""
    conf_frames = _confidence_frames_audio_v3(state)
    if int(state.frames) > 1:
        frame_pos = np.linspace(0.0, float(state.n_frames - 1), int(state.frames))
        return np.interp(np.arange(int(state.n_frames)), frame_pos, conf_frames).astype(np.float32)
    return np.full(int(state.n_frames), float(conf_frames[0]), dtype=np.float32)


def _render_audio_field_v3(
    state: _AudioFieldStateV3,
    *,
    coeff_vec: Optional[np.ndarray] = None,
    coarse_up: Optional[np.ndarray] = None,
    return_mask: bool = False,
):
    """Render int16 PCM samples from a v3 audio field state."""
    coeff_src = state.coeff_vec if coeff_vec is None else np.asarray(coeff_vec)
    coarse_src = state.coarse_up if coarse_up is None else np.asarray(coarse_up)

    ch = int(state.ch)
    n_frames = int(state.n_frames)
    frames = int(state.frames)
    bins = int(state.bins)
    n_fft = int(state.n_fft)
    hop = int(state.hop)

    total_expected = ch * frames * bins * 2
    coeff_use = np.asarray(coeff_src[:total_expected], dtype=np.float32, order="C")
    coeff_use = coeff_use.reshape(ch, frames, bins * 2)

    window = np.sqrt(np.hanning(n_fft)).astype(np.float32)
    steps = _audio_quant_steps(bins, int(state.quality), n_fft)

    residual = np.zeros((n_frames, ch), dtype=np.float64)
    for ch_idx in range(ch):
        q_re = coeff_use[ch_idx, :, 0::2].astype(np.float32)
        q_im = coeff_use[ch_idx, :, 1::2].astype(np.float32)
        spec = (q_re + 1j * q_im) * steps[None, :]
        spec *= float(n_fft)
        res_ch = _istft_1d(spec, n_fft, hop, window, n_frames)
        residual[:, ch_idx] = res_ch

    recon = coarse_src.astype(np.int32) + np.round(residual).astype(np.int32)
    recon = np.clip(recon, -32768, 32767).astype(np.int16)

    if return_mask:
        return recon, _confidence_curve_audio_v3(state)
    return recon


def _write_audio_field_v3(
    state: _AudioFieldStateV3,
    out_dir: str,
    *,
    target_chunk_kb: Optional[int] = None,
    block_count: Optional[int] = None,
) -> None:
    """Write a new v3 audio `.holo` directory from an existing v3 field state."""
    coeff = np.asarray(state.coeff_vec)
    if coeff.dtype != np.int16:
        coeff = np.rint(coeff).astype(np.int64)
        coeff = np.clip(coeff, -32768, 32767).astype(np.int16)

    residual_bytes_total = int(coeff.size) * 2
    coarse_payload = bytes(state.coarse_payload)

    B = int(block_count) if block_count is not None else int(state.block_count)
    if target_chunk_kb is not None:
        B = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    B = max(1, min(int(B), max(1, coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(coeff.size) if B > 1 else None
    chunks: list[tuple[float, int, bytes]] = []
    entries: list[dict] = []

    for block_id in range(B):
        if perm is None or B == 1:
            vals = coeff[block_id::B]
        else:
            idx = perm[block_id::B]
            vals = coeff[idx]
        vals_bytes = vals.astype("<i2", copy=False).tobytes()
        resid_comp = zlib.compress(vals_bytes, level=9)
        score = float(np.sum(vals.astype(np.float64) ** 2))
        chunks.append((score, int(block_id), resid_comp))

    chunks.sort(key=lambda x: x[0], reverse=True)

    for out_idx, (score, block_id, resid_comp) in enumerate(chunks):
        header = bytearray()
        header += MAGIC_AUD
        header += struct.pack("B", int(VERSION_AUD_OLO))
        header += struct.pack("B", int(state.ch))
        header += struct.pack("B", 2)
        header += struct.pack("B", int(state.flags))
        header += struct.pack(">I", int(state.sr))
        header += struct.pack(">I", int(state.n_frames))
        header += struct.pack(">I", int(B))
        header += struct.pack(">I", int(block_id))
        header += struct.pack(">I", int(state.coarse_len))
        header += struct.pack(">I", int(len(coarse_payload)))
        header += struct.pack(">I", int(len(resid_comp)))

        chunk_path = os.path.join(out_dir, f"chunk_{out_idx:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(bytes(header) + coarse_payload + resid_comp)

        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{score:.6f}")
        except OSError:
            pass
        entries.append({"file": os.path.basename(chunk_path), "block_id": int(block_id), "score": float(score)})

    if entries:
        entries.sort(key=lambda e: float(e.get("score", 0.0)), reverse=True)
        _write_chunk_manifest(
            out_dir,
            base_kind="audio",
            codec_version=int(VERSION_AUD_OLO),
            block_count=int(B),
            entries=entries,
        )


def _decode_audio_holo_core(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_mask: bool = False,
) -> Tuple[np.ndarray, int]:
    chunk_files = _select_chunk_files(in_dir, max_chunks=None, prefer_gain=False)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")

    for path in chunk_files:
        try:
            with open(path, "rb") as f:
                data = f.read(AUD_HEADER_SIZE)
        except OSError:
            continue
        try:
            version, *_ = _parse_audio_header(data)
        except ChunkFormatError:
            continue
        if version == VERSION_AUD_OLO:
            return _decode_audio_holo_core_v3(
                in_dir,
                max_chunks=max_chunks,
                prefer_gain=prefer_gain,
                use_recovery=use_recovery,
                return_mask=return_mask,
            )
        return _decode_audio_holo_core_v1_v2(
            in_dir,
            max_chunks=max_chunks,
            prefer_gain=prefer_gain,
            return_mask=return_mask,
        )

    raise ValueError(f"No decodable audio chunks found in {in_dir}")


def decode_audio_holo_dir(
    in_dir: str,
    output_wav: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
) -> None:
    audio, sr = _decode_audio_holo_core(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
    )
    _write_wav_int16(output_wav, audio, int(sr))


def decode_audio_holo_dir_meta(
    in_dir: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
    return_sr: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode audio and return (samples, confidence_curve).
    """
    audio, sr, conf = _decode_audio_holo_core(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
        return_mask=True,
    )
    if return_sr:
        return audio, int(sr), conf
    return audio, conf


def decode_audio_olonomic_holo_dir(
    in_dir: str,
    output_wav: str,
    *,
    max_chunks: Optional[int] = None,
    prefer_gain: bool = False,
    use_recovery: Optional[bool] = None,
) -> None:
    audio, sr = _decode_audio_holo_core_v3(
        in_dir,
        max_chunks=max_chunks,
        prefer_gain=prefer_gain,
        use_recovery=use_recovery,
    )
    _write_wav_int16(output_wav, audio, int(sr))


# ===================== Gauge alignment + stacking (v3 audio) =====================


def _phase_corr_shift1d(ref: np.ndarray, tgt: np.ndarray) -> tuple[int, float]:
    """
    Estimate integer shift to apply to tgt so it aligns to ref (phase correlation).
    """
    a = np.asarray(ref, dtype=np.float32).reshape(-1)
    b = np.asarray(tgt, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0, 0.0

    n = int(min(a.size, b.size))
    a = a[:n]
    b = b[:n]

    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    if n > 1:
        w = np.hanning(n).astype(np.float32)
        a = a * w
        b = b * w

    A = np.fft.fft(a)
    B = np.fft.fft(b)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-12)
    corr = np.fft.ifft(R).real

    i = int(np.argmax(corr))
    shift = i if i <= n // 2 else i - n
    peak = float(corr[i])
    return int(shift), float(peak)


def _audio_envelope_frames_v3(state: _AudioFieldStateV3) -> np.ndarray:
    """
    Build a low-rate envelope (length = STFT frames) from the coarse waveform.
    """
    coarse = np.asarray(state.coarse_up, dtype=np.float32)
    if coarse.ndim == 2 and int(state.ch) > 1:
        mono = np.mean(np.abs(coarse), axis=1)
    elif coarse.ndim == 2:
        mono = np.abs(coarse[:, 0])
    else:
        mono = np.abs(coarse)

    frames = int(state.frames)
    n_fft = int(state.n_fft)
    hop = int(state.hop)
    n_samples = int(state.n_frames)

    if frames <= 0:
        return np.zeros(0, dtype=np.float32)

    env = np.zeros(frames, dtype=np.float32)
    for fi in range(frames):
        t0 = fi * hop
        t1 = min(n_samples, t0 + n_fft)
        if t1 <= t0:
            continue
        env[fi] = float(np.mean(mono[t0:t1]))

    env -= float(np.mean(env)) if env.size > 0 else 0.0
    if env.size > 1:
        env *= np.hanning(int(env.size)).astype(np.float32)
    return env


def _estimate_frame_shift_audio_v3(
    ref: _AudioFieldStateV3,
    tgt: _AudioFieldStateV3,
    *,
    min_peak: float = 0.15,
    max_shift_frames: Optional[int] = None,
) -> tuple[int, float]:
    """
    Estimate an integer frame shift to apply to tgt so it aligns to ref.
    """
    env_ref = _audio_envelope_frames_v3(ref)
    env_tgt = _audio_envelope_frames_v3(tgt)
    if env_ref.size == 0 or env_tgt.size == 0 or env_ref.size != env_tgt.size:
        return 0, 0.0

    shift, peak = _phase_corr_shift1d(env_ref, env_tgt)
    if not np.isfinite(peak) or peak < float(min_peak):
        return 0, float(peak)

    lim = env_ref.size // 2
    if max_shift_frames is not None:
        lim = min(lim, max(0, int(max_shift_frames)))
    shift = int(np.clip(int(shift), -int(lim), int(lim)))
    return int(shift), float(peak)


def _shift_audio_samples_xy(arr: np.ndarray, shift_samples: int) -> np.ndarray:
    """
    Shift a (n_samples, ch) array with zero fill.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[:, None]
    n = int(a.shape[0])
    sy0, sy1, dy0, dy1 = _shift_slices_1d(n, int(shift_samples))
    out = np.zeros_like(a)
    if sy1 <= sy0 or dy1 <= dy0:
        return out
    out[dy0:dy1, :] = a[sy0:sy1, :]
    return out


def _shift_audio_coeff_frames_v3(
    state: _AudioFieldStateV3,
    shift_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift a v3 audio coefficient tensor by whole STFT frames.
    """
    ch = int(state.ch)
    frames = int(state.frames)
    bins = int(state.bins)
    total_expected = ch * frames * bins * 2

    coeff_src = np.asarray(state.coeff_vec[:total_expected], dtype=np.int16, order="C")
    coeff_src = coeff_src.reshape(ch, frames, bins * 2)

    sy0, sy1, dy0, dy1 = _shift_slices_1d(frames, int(shift_frames))
    coeff_out = np.zeros_like(coeff_src)
    if sy1 > sy0 and dy1 > dy0:
        coeff_out[:, dy0:dy1, :] = coeff_src[:, sy0:sy1, :]

    if state.coeff_mask is not None:
        mask_src = np.asarray(state.coeff_mask[:total_expected], dtype=bool, order="C")
        mask_src = mask_src.reshape(ch, frames, bins * 2)
        mask_out = np.zeros_like(mask_src)
        if sy1 > sy0 and dy1 > dy0:
            mask_out[:, dy0:dy1, :] = mask_src[:, sy0:sy1, :]
        mask_flat = mask_out.reshape(-1)
    else:
        frame_mask = np.zeros(frames, dtype=bool)
        if dy1 > dy0:
            frame_mask[dy0:dy1] = True
        per_frame = np.repeat(frame_mask, bins * 2)
        mask_flat = np.tile(per_frame, ch).astype(bool)

    coeff_flat = coeff_out.reshape(-1)
    coeff_full = np.zeros_like(state.coeff_vec, dtype=np.int16)
    mask_full = np.zeros_like(state.coeff_vec, dtype=bool)
    coeff_full[:total_expected] = coeff_flat
    mask_full[:total_expected] = mask_flat
    return coeff_full, mask_full


def stack_audio_holo_dirs(
    in_dirs: Sequence[str],
    output_wav: str,
    *,
    max_chunks: Optional[int] = None,
    gauge_align: bool = True,
    min_gauge_peak: float = 0.15,
    max_shift_frames: Optional[int] = None,
) -> None:
    """
    Stack multiple holographic audio directories.

    For mixed / legacy fields we fall back to sample-domain averaging.

    For v3 audio fields we stack in coefficient space. When `gauge_align` is on,
    we estimate an integer frame shift between exposures from the coarse
    waveforms and shift the coefficient tensors before summing.
    """
    v3_states: list[_AudioFieldStateV3] = []
    all_v3 = True

    for d in in_dirs:
        if not os.path.isdir(d):
            continue
        chunk_files = sorted(glob.glob(os.path.join(d, "chunk_*.holo")))
        if not chunk_files:
            continue
        try:
            with open(chunk_files[0], "rb") as f:
                head = f.read(AUD_HEADER_SIZE)
            version, *_ = _parse_audio_header(head)
        except Exception:
            all_v3 = False
            break
        if version != VERSION_AUD_OLO:
            all_v3 = False
            break

        try:
            st = _load_audio_field_v3(
                d,
                max_chunks=max_chunks,
                prefer_gain=False,
                use_recovery=None,
                return_mask=True,
            )
            v3_states.append(st)
        except Exception:
            all_v3 = False
            break

    if all_v3 and v3_states:
        ref = v3_states[0]
        total_coeff = int(ref.coeff_vec.size)

        coeff_sum = np.zeros(total_coeff, dtype=np.float64)
        coeff_w = np.zeros(total_coeff, dtype=np.float64)

        coarse_sum = np.zeros((int(ref.n_frames), int(ref.ch)), dtype=np.float64)
        coarse_w = np.zeros((int(ref.n_frames),), dtype=np.float64)

        for st in v3_states:
            if (st.sr, st.ch, st.n_frames, st.n_fft, st.hop, st.frames, st.bins) != (
                ref.sr, ref.ch, ref.n_frames, ref.n_fft, ref.hop, ref.frames, ref.bins
            ):
                continue
            if int(st.coeff_vec.size) != total_coeff:
                continue

            shift_frames = 0
            if gauge_align and st is not ref:
                try:
                    shift_frames, _peak = _estimate_frame_shift_audio_v3(
                        ref,
                        st,
                        min_peak=float(min_gauge_peak),
                        max_shift_frames=max_shift_frames,
                    )
                except Exception:
                    shift_frames = 0

            shift_samples = int(shift_frames) * int(ref.hop)

            coeff_shift, mask_shift = _shift_audio_coeff_frames_v3(st, int(shift_frames))
            coarse_shift = _shift_audio_samples_xy(np.asarray(st.coarse_up, dtype=np.float32), shift_samples)
            valid = _shift_audio_samples_xy(
                np.ones((int(ref.n_frames), 1), dtype=np.float32),
                shift_samples,
            )[:, 0]

            coarse_sum += coarse_shift.astype(np.float64) * valid[:, None].astype(np.float64)
            coarse_w += valid.astype(np.float64)

            m = np.asarray(mask_shift, dtype=bool)
            coeff_sum[m] += coeff_shift[m].astype(np.float64)
            coeff_w[m] += 1.0

        if float(np.max(coarse_w)) <= 0.0:
            raise ValueError("No compatible v3 audio fields found to stack")

        coarse_avg = (coarse_sum / np.maximum(coarse_w[:, None], 1.0)).astype(np.float32)

        coeff_avg = np.zeros(total_coeff, dtype=np.float32)
        np.divide(coeff_sum, np.maximum(coeff_w, 1.0), out=coeff_avg, where=coeff_w > 0.0)

        audio = _render_audio_field_v3(ref, coeff_vec=coeff_avg, coarse_up=coarse_avg)
        _write_wav_int16(output_wav, audio, int(ref.sr))
        return

    acc = None
    count = 0
    sr_ref = None

    for d in in_dirs:
        if not os.path.isdir(d):
            continue
        try:
            audio, sr = _decode_audio_holo_core(d, max_chunks=max_chunks)
        except Exception:
            continue
        if sr_ref is None:
            sr_ref = int(sr)
        if int(sr) != int(sr_ref):
            continue
        a = np.asarray(audio, dtype=np.float32)
        if acc is None:
            acc = a
        else:
            n = min(int(acc.shape[0]), int(a.shape[0]))
            if n <= 0:
                continue
            acc = acc[:n] + a[:n]
        count += 1

    if acc is None or count <= 0 or sr_ref is None:
        raise ValueError("No decodable audio holographic directories to stack")

    out = np.clip(np.rint(acc / float(count)), -32768, 32767).astype(np.int16)
    _write_wav_int16(output_wav, out, int(sr_ref))




# ===================== Mode detection helpers =====================

def detect_mode_from_extension(path: str) -> str:
    """
    Detect whether a path should be treated as 'image' or 'audio' based on its extension.

    Currently:
      - '.wav'  -> 'audio'
      - anything else -> 'image'
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return "audio"
    return "image"


def detect_mode_from_chunk_dir(in_dir: str) -> str:
    """
    Detect chunk type ('image' or 'audio') by inspecting the first chunk magic bytes.

    Parameters
    ----------
    in_dir : str
        Directory containing chunk_*.holo files.

    Returns
    -------
    str
        'image' if the magic is MAGIC_IMG, 'audio' if MAGIC_AUD.

    Raises
    ------
    FileNotFoundError
        If no chunk files are found.
    ValueError
        If the magic is neither image nor audio.
    """
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
