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

import glob
import math
import os
import struct
import wave
import zlib
from io import BytesIO
from typing import Optional, Tuple, Sequence

import numpy as np
from PIL import Image


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
    Remove existing chunk_*.holo files from a directory.

    This is used before a new encode so that stale chunks from a previous
    encode do not accidentally "mix" into a new holographic field.
    """
    for p in glob.glob(os.path.join(out_dir, "chunk_*.holo")):
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


def encode_image_olonomic_holo_dir(
    input_path: str,
    out_dir: str,
    *,
    block_count: int = 32,
    coarse_max_side: int = 16,
    target_chunk_kb: Optional[int] = None,
    max_chunk_bytes: Optional[int] = None,
    quality: int = 50,
    dct_block: int = 8,
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

    # Coarse thumbnail identical to legacy path (always stored as PNG)
    max_side = max(h, w)
    scale = min(1.0, float(coarse_max_side) / float(max_side))
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))

    img_pil = Image.fromarray(img, mode="RGB")
    coarse_img = img_pil.resize((cw, ch), resample=_BICUBIC)

    buf = BytesIO()
    coarse_img.save(buf, format="PNG")
    coarse_png = buf.getvalue()

    coarse_up = coarse_img.resize((w, h), resample=_BICUBIC)
    coarse_up_arr = np.asarray(coarse_up, dtype=np.uint8).astype(np.int16, copy=False)

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

    coarse_payload = OLOI_META.pack(
        OLOI_MAGIC,
        1,  # meta version
        int(block),
        quality_u8,
        OLOI_FLAG_COEFF_CLIPPED if clipped else 0,
        int(pad_h),
        int(pad_w),
        int(len(coarse_png)),
    ) + coarse_png

    if max_chunk_bytes is not None:
        block_count = _select_block_count_bytes(residual_bytes_total, len(coarse_payload), int(max_chunk_bytes))
    elif target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), max(1, residual_coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(residual_coeff.size) if block_count > 1 else None

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_coeff[block_id::block_count]
        else:
            idx = perm[block_id::block_count]
            vals = residual_coeff[idx]

        resid_comp = zlib.compress(vals.astype("<i2", copy=False).tobytes(), level=9)

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

        gain = float(np.sum(np.abs(vals), dtype=np.float64))
        meta_path = chunk_path + ".meta"
        try:
            with open(meta_path, "w", encoding="ascii") as mf:
                mf.write(f"{gain:.6f}")
        except OSError:
            pass


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


def _decode_image_holo_core_v1_v2(in_dir: str, *, max_chunks: Optional[int] = None) -> np.ndarray:
    """
    Internal decoder for image versions 1/2 that reconstructs an RGB image from a .holo directory.

    It reads up to max_chunks chunks (if given), or all available chunks otherwise.
    For each chunk it:
      - parses the header,
      - reconstructs / reuses the coarse thumbnail,
      - decompresses and places its residual slice into the global residual vector.

    This function returns an RGB uint8 array and does not write to disk.
    """
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

        if block_count == 1:
            # Simple case: one chunk carries the entire residual
            residual_flat[: min(residual_flat.size, vals.size)] = vals[: residual_flat.size]
            continue

        if version_used == VERSION_IMG and perm is not None:
            idx = perm[block_id::block_count]
            n = min(idx.size, vals.size)
            residual_flat[idx[:n]] = vals[:n]
        else:
            # Legacy v1 layout: plain striding
            pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
            pos = pos[pos < residual_flat.size]
            residual_flat[pos] = vals[: pos.size]

    if residual_flat is None or coarse_up_arr is None:
        raise ValueError(f"No decodable image chunks found in {in_dir}")

    recon = coarse_up_arr + residual_flat.reshape(int(h), int(w), int(c))
    return np.clip(recon, 0, 255).astype(np.uint8)


def _decode_image_holo_core_v3(in_dir: str, *, max_chunks: Optional[int] = None) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")
    if max_chunks is not None:
        chunk_files = chunk_files[: max(1, int(max_chunks))]

    h = w = c = None
    block_count = None
    block_size = None
    pad_h = pad_w = None
    quality = None
    coarse_up_arr = None
    coeff_vec = None
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
                magic, meta_ver, block_size_i, quality_i, flags, pad_h_i, pad_w_i, png_len = OLOI_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOI_MAGIC or meta_ver != 1:
                continue
            if block_size_i <= 0:
                continue
            if png_len > len(coarse_payload) - OLOI_META.size:
                continue
            png_bytes = coarse_payload[OLOI_META.size: OLOI_META.size + png_len]
            try:
                with Image.open(BytesIO(png_bytes)) as cim:
                    cim = cim.convert("RGB")
                    coarse_up = cim.resize((w_i, h_i), resample=_BICUBIC)
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

        if block_count == 1 or perm is None:
            pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
            pos = pos[pos < coeff_vec.size]
            coeff_vec[pos] = vals[: pos.size]
        else:
            idx = perm[block_id::block_count]
            n = min(idx.size, vals.size)
            coeff_vec[idx[:n]] = vals[:n]

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
    return np.clip(recon, 0, 255).astype(np.uint8)


def _decode_image_holo_core(in_dir: str, *, max_chunks: Optional[int] = None) -> np.ndarray:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
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
            return _decode_image_holo_core_v3(in_dir, max_chunks=max_chunks)
        return _decode_image_holo_core_v1_v2(in_dir, max_chunks=max_chunks)

    raise ValueError(f"No decodable image chunks found in {in_dir}")


def decode_image_holo_dir(in_dir: str, output_path: str, *, max_chunks: Optional[int] = None) -> None:
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
    recon = _decode_image_holo_core(in_dir, max_chunks=max_chunks)
    save_image_rgb_u8(recon, output_path)


def decode_image_olonomic_holo_dir(in_dir: str, output_path: str, *, max_chunks: Optional[int] = None) -> None:
    """
    Decode an olonomic (version 3) holographic image directory.
    """
    recon = _decode_image_holo_core_v3(in_dir, max_chunks=max_chunks)
    save_image_rgb_u8(recon, output_path)


def stack_image_holo_dirs(
    in_dirs: Sequence[str],
    output_path: str,
    *,
    max_chunks: Optional[int] = None,
) -> None:
    """
    Stack multiple holographic image directories to improve SNR over time.

    This function implements a simple "photon collector" style behavior:
    several `.holo` directories that encode the same scene (for example,
    repeated noisy captures of a faint object) are decoded, converted to
    float32, summed and averaged pixel-wise.

    The result is an image with the same mean brightness but reduced
    uncorrelated noise, exactly as in classical exposure stacking in
    astrophotography.

    Parameters
    ----------
    in_dirs : sequence of str
        A list of `.holo` directories that should all decode to images with
        the same geometry (H, W, 3). Incompatible directories are skipped.
    output_path : str
        Path where the stacked RGB uint8 image will be written.
    max_chunks : int, optional
        Upper bound on chunks used from each directory. This lets you simulate
        stacking under partial coverage, e.g. limited chunks per time window.

    Example
    -------
    Stack three holographic exposures of the same scene:

        stack_image_holo_dirs(
            ["t0.png.holo", "t1.png.holo", "t2.png.holo"],
            "frame_stacked.png",
        )

    This is a purely phenotypic operation: it works on reconstructed images,
    not on raw chunks. If you want to merge chunks into a new .holo field,
    that should be done at a higher layer (field/cortex/mesh).
    """
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
                # Skip directories with different geometry
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
    target_chunk_kb: Optional[int] = None,
    quality: int = 50,
    n_fft: int = 512,
    hop: Optional[int] = None,
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

    # Coarse envelope identical to legacy path
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

    coarse_zlib = zlib.compress(coarse.astype("<i2", copy=False).tobytes(), level=9)
    meta_flags = OLOA_FLAG_COEFF_CLIPPED if clipped else 0
    coarse_payload = OLOA_META.pack(
        OLOA_MAGIC,
        1,  # meta version
        quality_u8,
        meta_flags,
        0,
        int(n_fft),
        int(hop_val),
    ) + coarse_zlib

    if target_chunk_kb is not None:
        block_count = _select_block_count(residual_bytes_total, len(coarse_payload), int(target_chunk_kb))

    block_count = max(1, min(int(block_count), max(1, residual_coeff.size)))

    os.makedirs(out_dir, exist_ok=True)
    _wipe_old_chunks(out_dir)

    perm = _golden_permutation(residual_coeff.size) if block_count > 1 else None

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

        resid_comp = zlib.compress(vals.astype("<i2", copy=False).tobytes(), level=9)
        gain = float(np.sum(np.abs(vals), dtype=np.float64))
        chunks.append((gain, block_id, resid_comp))

    chunks.sort(key=lambda x: x[0], reverse=True)

    for out_idx, (gain, block_id, resid_comp) in enumerate(chunks):
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
                mf.write(f"{gain:.6f}")
        except OSError:
            pass


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


def _decode_audio_holo_core_v1_v2(in_dir: str, *, max_chunks: Optional[int] = None) -> Tuple[np.ndarray, int]:
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
    return recon, int(sr)


def _decode_audio_holo_core_v3(in_dir: str, *, max_chunks: Optional[int] = None) -> Tuple[np.ndarray, int]:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo found in {in_dir}")
    if max_chunks is not None:
        chunk_files = chunk_files[: max(1, int(max_chunks))]

    sr = ch = n_frames = None
    block_count = None
    coarse_len = None
    n_fft = None
    hop = None
    quality = None
    coeff_vec = None
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
                magic, meta_ver, quality_i, meta_flags, _reserved, n_fft_i, hop_i = OLOA_META.unpack_from(coarse_payload, 0)
            except struct.error:
                continue
            if magic != OLOA_MAGIC or meta_ver != 1:
                continue

            coarse_zlib = coarse_payload[OLOA_META.size:]
            try:
                coarse_bytes = zlib.decompress(coarse_zlib)
                coarse = np.frombuffer(coarse_bytes, dtype="<i2").astype(np.int16, copy=False)
                coarse = coarse.reshape(int(coarse_len_i), int(ch_i))
            except Exception:
                continue

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

            t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
            k0 = np.floor(t).astype(np.int64)
            k1 = np.clip(k0 + 1, 0, coarse_len - 1)
            alpha = (t - k0).astype(np.float64)[:, None]
            coarse_f = coarse.astype(np.float64)
            coarse_up_f = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
            coarse_up = np.round(coarse_up_f).astype(np.int16)

            frames, _ = _stft_frame_count(n_frames, n_fft, hop)
            bins = n_fft // 2 + 1
            total_coeff = int(ch) * frames * bins * 2
            coeff_vec = np.zeros(total_coeff, dtype=np.int16)

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

        if block_count == 1 or perm is None:
            pos = np.arange(block_id, block_id + vals.size * block_count, block_count, dtype=np.int64)
            pos = pos[pos < coeff_vec.size]
            coeff_vec[pos] = vals[: pos.size]
        else:
            idxb = perm[block_id::block_count]
            n = min(idxb.size, vals.size)
            coeff_vec[idxb[:n]] = vals[:n]
        used_chunks += 1

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
    return recon, int(sr)


def _decode_audio_holo_core(in_dir: str, *, max_chunks: Optional[int] = None) -> Tuple[np.ndarray, int]:
    chunk_files = sorted(glob.glob(os.path.join(in_dir, "chunk_*.holo")))
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
            return _decode_audio_holo_core_v3(in_dir, max_chunks=max_chunks)
        return _decode_audio_holo_core_v1_v2(in_dir, max_chunks=max_chunks)

    raise ValueError(f"No decodable audio chunks found in {in_dir}")


def decode_audio_holo_dir(in_dir: str, output_wav: str, *, max_chunks: Optional[int] = None) -> None:
    audio, sr = _decode_audio_holo_core(in_dir, max_chunks=max_chunks)
    _write_wav_int16(output_wav, audio, int(sr))


def decode_audio_olonomic_holo_dir(in_dir: str, output_wav: str, *, max_chunks: Optional[int] = None) -> None:
    audio, sr = _decode_audio_holo_core_v3(in_dir, max_chunks=max_chunks)
    _write_wav_int16(output_wav, audio, int(sr))




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
