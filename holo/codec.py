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

# Golden ratio constant used for the permutation step.
PHI = (1.0 + 5.0 ** 0.5) / 2.0

# Fixed-size header layouts.
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


def encode_image_holo_dir(
    input_path: str,
    out_dir: str,
    *,
    block_count: int = 32,
    coarse_max_side: int = 16,
    target_chunk_kb: Optional[int] = None,
    max_chunk_bytes: Optional[int] = None,
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
    coarse_img.save(buf, format="PNG")
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
    """
    Internal decoder that reconstructs an RGB image from a .holo directory.

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
    """
    Decode a holographic audio directory back into a 16-bit PCM WAV file.

    Parameters
    ----------
    in_dir : str
        Directory containing audio chunk_*.holo files.
    output_wav : str
        Output WAV file path.
    max_chunks : int, optional
        If given, only the first max_chunks chunks are used.

    Example
    -------
    Decode using all chunks:

        decode_audio_holo_dir("track.wav.holo", "track_recon.wav")
    """
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
    _write_wav_int16(output_wav, recon, int(sr))


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
