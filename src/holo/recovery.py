"""
holo.recovery

Systematic RLNC (GF(256)) helpers for optional recovery chunks.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


REC_MAGIC = b"HORC"
REC_VERSION = 1

REC_KIND_IMAGE = 1
REC_KIND_AUDIO = 2

REC_FLAG_ZLIB = 1 << 0

REC_HEADER = struct.Struct(">4sBBBBIIIII")


def _build_gf256_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exp = np.zeros(512, dtype=np.uint8)
    log = np.zeros(256, dtype=np.int16)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    exp[255:510] = exp[:255]
    exp[510:] = exp[:2]

    mul = np.zeros((256, 256), dtype=np.uint8)
    for a in range(256):
        if a == 0:
            continue
        la = int(log[a])
        for b in range(256):
            if b == 0:
                continue
            mul[a, b] = exp[la + int(log[b])]

    inv = np.zeros(256, dtype=np.uint8)
    inv[0] = 0
    for a in range(1, 256):
        inv[a] = exp[255 - int(log[a])]

    return exp, log, mul, inv


_EXP, _LOG, _MUL, _INV = _build_gf256_tables()


def gf256_mul_vec(coeff: int, vec: np.ndarray) -> np.ndarray:
    if coeff == 0:
        return np.zeros_like(vec, dtype=np.uint8)
    return _MUL[int(coeff), vec]


def gf256_gauss_jordan(A: np.ndarray, B: np.ndarray) -> tuple[Optional[np.ndarray], int]:
    """
    Solve A * X = B over GF(256) via Gauss-Jordan.

    Returns (X, rank). X is None when rank < n_cols.
    """
    A = np.array(A, dtype=np.uint8, copy=True)
    B = np.array(B, dtype=np.uint8, copy=True)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")
    m, n = A.shape
    if B.shape[0] != m:
        raise ValueError("A and B row counts must match")

    row = 0
    pivot_cols: list[int] = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
            B[[row, pivot]] = B[[pivot, row]]

        pv = int(A[row, col])
        if pv != 1:
            inv = int(_INV[pv])
            A[row] = _MUL[inv, A[row]]
            B[row] = _MUL[inv, B[row]]

        for r in range(m):
            if r == row:
                continue
            factor = int(A[r, col])
            if factor != 0:
                A[r] ^= _MUL[factor, A[row]]
                B[r] ^= _MUL[factor, B[row]]

        pivot_cols.append(col)
        row += 1
        if row >= m:
            break

    rank = row
    if rank < n:
        return None, rank

    X = np.zeros((n, B.shape[1]), dtype=np.uint8)
    for r, col in enumerate(pivot_cols):
        X[col] = B[r]
    return X, rank


@dataclass
class RecoveryChunk:
    base_kind: int
    base_codec_version: int
    block_count: int
    coded_id: int
    slice_len: int
    coeffs: np.ndarray
    payload: np.ndarray
    flags: int


def encode_recovery_chunk(
    *,
    base_kind: int,
    base_codec_version: int,
    block_count: int,
    coded_id: int,
    slice_len: int,
    coeffs: np.ndarray,
    payload: bytes,
    compress: bool = True,
) -> bytes:
    flags = 0
    payload_bytes = payload
    if compress:
        comp = zlib.compress(payload, level=9)
        if len(comp) < len(payload):
            payload_bytes = comp
            flags |= REC_FLAG_ZLIB

    header = REC_HEADER.pack(
        REC_MAGIC,
        REC_VERSION,
        int(base_kind),
        int(base_codec_version),
        int(flags),
        int(block_count),
        int(coded_id),
        int(slice_len),
        int(coeffs.size),
        int(len(payload_bytes)),
    )
    return header + coeffs.astype(np.uint8, copy=False).tobytes() + payload_bytes


def parse_recovery_chunk(data: bytes) -> Optional[RecoveryChunk]:
    if len(data) < REC_HEADER.size:
        return None
    try:
        (magic, rec_version, base_kind, base_codec_version, flags,
         block_count, coded_id, slice_len, coeff_len, payload_len) = REC_HEADER.unpack_from(data, 0)
    except struct.error:
        return None
    if magic != REC_MAGIC or rec_version != REC_VERSION:
        return None
    if coeff_len <= 0 or payload_len <= 0:
        return None
    off = REC_HEADER.size
    need = off + int(coeff_len) + int(payload_len)
    if need > len(data):
        return None
    coeff_bytes = data[off: off + int(coeff_len)]
    off += int(coeff_len)
    payload_bytes = data[off: off + int(payload_len)]
    if flags & REC_FLAG_ZLIB:
        try:
            payload_bytes = zlib.decompress(payload_bytes)
        except Exception:
            return None
    if slice_len != len(payload_bytes):
        return None
    coeffs = np.frombuffer(coeff_bytes, dtype=np.uint8).copy()
    payload = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
    return RecoveryChunk(
        base_kind=int(base_kind),
        base_codec_version=int(base_codec_version),
        block_count=int(block_count),
        coded_id=int(coded_id),
        slice_len=int(slice_len),
        coeffs=coeffs,
        payload=payload,
        flags=int(flags),
    )


def build_recovery_chunks(
    slices: Sequence[bytes],
    *,
    base_kind: int,
    base_codec_version: int,
    overhead: float,
    seed: Optional[int] = None,
) -> list[bytes]:
    block_count = len(slices)
    if block_count <= 0:
        return []
    if overhead <= 0.0:
        return []
    R = int(math.ceil(float(block_count) * float(overhead)))
    if R <= 0:
        return []

    lengths = [len(s) for s in slices]
    slice_len = max(lengths) if lengths else 0
    if slice_len <= 0:
        return []

    padded: list[np.ndarray] = []
    for s in slices:
        buf = np.zeros(slice_len, dtype=np.uint8)
        if s:
            arr = np.frombuffer(s, dtype=np.uint8)
            buf[: arr.size] = arr
        padded.append(buf)

    rng = np.random.default_rng(seed)
    chunks: list[bytes] = []

    for coded_id in range(R):
        coeffs = rng.integers(1, 256, size=block_count, dtype=np.uint8)
        acc = np.zeros(slice_len, dtype=np.uint8)
        for idx, coeff in enumerate(coeffs):
            if coeff == 0:
                continue
            acc ^= _MUL[int(coeff), padded[idx]]

        chunk_bytes = encode_recovery_chunk(
            base_kind=base_kind,
            base_codec_version=base_codec_version,
            block_count=block_count,
            coded_id=int(coded_id),
            slice_len=int(slice_len),
            coeffs=coeffs,
            payload=acc.tobytes(),
            compress=True,
        )
        chunks.append(chunk_bytes)

    return chunks


def recover_missing_slices(
    *,
    block_count: int,
    missing_ids: Sequence[int],
    known_slices: dict[int, bytes],
    recovery_chunks: Iterable[RecoveryChunk],
    slice_len: int,
) -> Optional[dict[int, bytes]]:
    if not missing_ids:
        return {}
    if slice_len <= 0:
        return None

    missing = [int(m) for m in missing_ids]
    if not missing:
        return {}

    known_padded: dict[int, np.ndarray] = {}
    for block_id, payload in known_slices.items():
        buf = np.zeros(slice_len, dtype=np.uint8)
        if payload:
            arr = np.frombuffer(payload, dtype=np.uint8)
            buf[: arr.size] = arr
        known_padded[int(block_id)] = buf

    A_rows: list[np.ndarray] = []
    B_rows: list[np.ndarray] = []

    for chunk in recovery_chunks:
        if chunk.block_count != int(block_count):
            continue
        if chunk.slice_len != int(slice_len):
            continue
        coeffs = chunk.coeffs
        if coeffs.size < block_count:
            continue
        rhs = np.array(chunk.payload, dtype=np.uint8, copy=True)

        for block_id, payload in known_padded.items():
            coeff = int(coeffs[block_id])
            if coeff != 0:
                rhs ^= _MUL[coeff, payload]

        row = coeffs[np.array(missing, dtype=np.int64)]
        if np.any(row):
            A_rows.append(row.astype(np.uint8, copy=False))
            B_rows.append(rhs)

    if not A_rows:
        return None
    A = np.stack(A_rows, axis=0)
    B = np.stack(B_rows, axis=0)

    if A.shape[0] < len(missing):
        return None

    X, rank = gf256_gauss_jordan(A, B)
    if X is None or rank < len(missing):
        return None

    recovered: dict[int, bytes] = {}
    for idx, block_id in enumerate(missing):
        recovered[int(block_id)] = X[idx].tobytes()
    return recovered
