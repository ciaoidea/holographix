"""
holo.container

Multi-object packing helpers for Holographix.

Several images or audio tracks can be packed into a single holographic
field so they share the same residual trajectory. Losing chunks then
reduces detail across the whole set instead of destroying one file
while leaving another untouched.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

try:  # Pillow >= 9
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover
    _BICUBIC = Image.BICUBIC

from .codec import (
    VERSION_AUD,
    VERSION_IMG,
    _golden_permutation,
    _select_block_count,
    _write_wav_int16,
    detect_mode_from_extension,
    load_image_rgb_u8,
    save_image_rgb_u8,
    _read_wav_as_int16,
)

CONTAINER_MAGIC = b"HOPK"  # HOlographix PacK
CONTAINER_VERSION = 1
CONTAINER_HEADER = struct.Struct(">4sBIIQI")

__all__ = ["pack_objects_holo_dir", "unpack_object_from_holo_dir"]


def _encode_image_object(path: str, coarse_max_side: int, version: int) -> Dict[str, object]:
    img = load_image_rgb_u8(path)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("Only RGB images are supported for packing")

    max_side = max(h, w)
    scale = min(1.0, float(coarse_max_side) / float(max_side))
    cw = max(1, int(round(w * scale)))
    ch = max(1, int(round(h * scale)))

    img_pil = Image.fromarray(img, mode="RGB")
    coarse_img = img_pil.resize((cw, ch), resample=_BICUBIC)

    coarse_arr = np.asarray(coarse_img, dtype=np.uint8)
    coarse_io = zlib.compress(coarse_arr.tobytes(), level=9)

    coarse_up = coarse_img.resize((w, h), resample=_BICUBIC)
    coarse_up_arr = np.asarray(coarse_up, dtype=np.uint8)

    residual = img.astype(np.int16) - coarse_up_arr.astype(np.int16)
    residual_flat = residual.reshape(-1)

    return {
        "mode": "image",
        "h": int(h),
        "w": int(w),
        "c": int(c),
        "coarse_comp": coarse_io,
        "coarse_shape": (ch, cw, 3),
        "residual": residual_flat,
        "version": int(version),
    }


def _encode_audio_object(path: str, coarse_max_frames: int, version: int) -> Dict[str, object]:
    audio, sr, ch = _read_wav_as_int16(path)
    n_frames = int(audio.shape[0])
    if n_frames < 2:
        raise ValueError("Audio too short to pack")

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
    residual = np.clip(diff32, -32768, 32767).astype(np.int16)

    coarse_comp = zlib.compress(coarse.astype("<i2", copy=False).tobytes(), level=9)

    return {
        "mode": "audio",
        "sr": int(sr),
        "ch": int(ch),
        "n_frames": int(n_frames),
        "coarse_comp": coarse_comp,
        "coarse_len": int(coarse_len),
        "residual": residual.reshape(-1),
        "version": int(version),
    }


def pack_objects_holo_dir(
    paths: Sequence[str],
    out_dir: str,
    *,
    target_chunk_kb: Optional[int] = 32,
    block_count: Optional[int] = None,
    coarse_max_side: int = 64,
    coarse_max_frames: int = 2048,
) -> None:
    """
    Pack several images or WAV files into a single holographic directory.
    """
    if not paths:
        raise ValueError("No input paths provided")

    encoded: List[Dict[str, object]] = []
    offsets: List[int] = []
    total_residual = 0

    for path in paths:
        mode = detect_mode_from_extension(path)
        if mode == "image":
            obj = _encode_image_object(path, coarse_max_side, VERSION_IMG)
        else:
            obj = _encode_audio_object(path, coarse_max_frames, VERSION_AUD)

        offsets.append(total_residual)
        total_residual += int(obj["residual"].size)  # type: ignore[index]
        encoded.append(obj)

    if total_residual <= 0:
        raise ValueError("Nothing to pack (empty residuals)")

    if block_count is None:
        coarse_total = sum(len(obj["coarse_comp"]) for obj in encoded)  # type: ignore[index]
        block_count = _select_block_count(total_residual * 2, coarse_total, target_chunk_kb or 32)

    block_count = max(1, min(int(block_count), total_residual))
    perm = _golden_permutation(total_residual) if block_count > 1 else None

    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        "version": CONTAINER_VERSION,
        "block_count": int(block_count),
        "total_residual": int(total_residual),
        "objects": [],
    }

    for idx, (obj, off) in enumerate(zip(encoded, offsets)):
        coarse_name = f"obj_{idx:04d}_coarse.bin"
        coarse_path = os.path.join(out_dir, coarse_name)
        with open(coarse_path, "wb") as f:
            f.write(obj["coarse_comp"])  # type: ignore[index]

        entry: Dict[str, object] = {
            "mode": obj["mode"],
            "offset": int(off),
            "residual_len": int(obj["residual"].size),  # type: ignore[index]
            "coarse_file": coarse_name,
            "version": int(obj["version"]),
        }

        if obj["mode"] == "image":
            entry.update(
                {
                    "h": int(obj["h"]),
                    "w": int(obj["w"]),
                    "c": int(obj["c"]),
                    "coarse_shape": obj["coarse_shape"],
                }
            )
        else:
            entry.update(
                {
                    "sr": int(obj["sr"]),
                    "ch": int(obj["ch"]),
                    "n_frames": int(obj["n_frames"]),
                    "coarse_len": int(obj["coarse_len"]),
                }
            )

        manifest["objects"].append(entry)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    residual_all = np.zeros(total_residual, dtype=np.int16)
    for obj, off in zip(encoded, offsets):
        r = obj["residual"]  # type: ignore[index]
        residual_all[off:off + r.size] = r

    for block_id in range(block_count):
        if perm is None or block_count == 1:
            vals = residual_all[block_id::block_count]
        else:
            idxs = perm[block_id::block_count]
            vals = residual_all[idxs]

        resid_comp = zlib.compress(vals.astype("<i2", copy=False).tobytes(), level=9)
        header = CONTAINER_HEADER.pack(
            CONTAINER_MAGIC,
            CONTAINER_VERSION,
            int(block_count),
            int(block_id),
            int(total_residual),
            int(len(resid_comp)),
        )

        chunk_path = os.path.join(out_dir, f"chunk_{block_id:04d}.holo")
        with open(chunk_path, "wb") as f:
            f.write(header + resid_comp)


def _load_manifest(in_dir: str) -> Dict[str, object]:
    manifest_path = os.path.join(in_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def _parse_container_chunk(data: bytes) -> Optional[tuple]:
    if len(data) < CONTAINER_HEADER.size:
        return None
    magic, version, block_count, block_id, total_residual, resid_size = CONTAINER_HEADER.unpack_from(data, 0)
    if magic != CONTAINER_MAGIC or version != CONTAINER_VERSION:
        return None
    if block_count <= 0 or block_id < 0 or block_id >= block_count:
        return None
    if CONTAINER_HEADER.size + resid_size > len(data):
        return None
    resid_comp = data[CONTAINER_HEADER.size: CONTAINER_HEADER.size + resid_size]
    return block_count, block_id, total_residual, resid_comp


def unpack_object_from_holo_dir(
    in_dir: str,
    index: int,
    output_path: str,
    *,
    max_chunks: Optional[int] = None,
) -> None:
    """
    Reconstruct one object from a packed holographic directory.
    """
    manifest = _load_manifest(in_dir)
    objects = manifest.get("objects", [])
    if not objects or index < 0 or index >= len(objects):
        raise IndexError("Object index out of range for packed field")

    entry = objects[index]
    block_count = int(manifest["block_count"])
    total_residual = int(manifest["total_residual"])
    residual_flat = np.zeros(total_residual, dtype=np.int16)
    perm = _golden_permutation(total_residual) if block_count > 1 else None

    chunk_files = sorted(
        p for p in os.listdir(in_dir)
        if p.startswith("chunk_") and p.endswith(".holo")
    )
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.holo in {in_dir}")
    if max_chunks is not None:
        chunk_files = chunk_files[: max(1, int(max_chunks))]

    for name in chunk_files:
        path = os.path.join(in_dir, name)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            continue
        parsed = _parse_container_chunk(data)
        if parsed is None:
            continue
        B_i, block_id, tot_i, resid_comp = parsed
        if (B_i != block_count) or (tot_i != total_residual):
            continue

        try:
            vals_bytes = zlib.decompress(resid_comp)
            vals = np.frombuffer(vals_bytes, dtype="<i2").astype(np.int16, copy=False)
        except Exception:
            continue

        if block_count == 1:
            n = min(residual_flat.size, vals.size)
            residual_flat[:n] = vals[:n]
            continue

        if perm is not None:
            idxs = perm[block_id::block_count]
            n = min(idxs.size, vals.size)
            residual_flat[idxs[:n]] = vals[:n]

    off = int(entry["offset"])
    length = int(entry["residual_len"])
    if off + length > residual_flat.size:
        raise ValueError("Residual slice outside global buffer")
    slice_vals = residual_flat[off: off + length]

    coarse_path = os.path.join(in_dir, entry["coarse_file"])
    with open(coarse_path, "rb") as f:
        coarse_bytes = f.read()

    if entry["mode"] == "image":
        h = int(entry["h"]); w = int(entry["w"]); c = int(entry["c"])
        coarse_shape = tuple(entry["coarse_shape"])
        coarse_arr = np.frombuffer(zlib.decompress(coarse_bytes), dtype=np.uint8)
        coarse_arr = coarse_arr.reshape(coarse_shape)
        coarse_img = Image.fromarray(coarse_arr, mode="RGB")
        coarse_up = coarse_img.resize((w, h), resample=_BICUBIC)
        coarse_up_arr = np.asarray(coarse_up, dtype=np.int16)

        resid_arr = slice_vals.reshape(h, w, c)
        recon = coarse_up_arr + resid_arr
        save_image_rgb_u8(np.clip(recon, 0, 255).astype(np.uint8), output_path)
    else:
        sr = int(entry["sr"]); ch = int(entry["ch"])
        n_frames = int(entry["n_frames"]); coarse_len = int(entry["coarse_len"])
        coarse = np.frombuffer(zlib.decompress(coarse_bytes), dtype="<i2").astype(np.int16, copy=False)
        coarse = coarse.reshape(coarse_len, ch)

        t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
        k0 = np.floor(t).astype(np.int64)
        k1 = np.clip(k0 + 1, 0, coarse_len - 1)
        alpha = (t - k0).astype(np.float64)[:, None]
        coarse_f = coarse.astype(np.float64)
        coarse_up = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
        coarse_up = np.round(coarse_up).astype(np.int16)

        resid = slice_vals.reshape(n_frames, ch)
        recon = coarse_up.astype(np.int32) + resid.astype(np.int32)
        recon = np.clip(recon, -32768, 32767).astype(np.int16)
        _write_wav_int16(output_path, recon, sr)
