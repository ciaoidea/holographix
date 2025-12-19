"""
holo.tnc.afsk

Minimal AFSK modem (Bell-202 style) for TNC experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np

from .frame import FrameDecoder, build_frame


def bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8, copy=False)


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    if bits is None:
        return b""
    bits_arr = np.array(bits, dtype=np.uint8, copy=False)
    if bits_arr.size % 8 != 0:
        bits_arr = bits_arr[: bits_arr.size - (bits_arr.size % 8)]
    if bits_arr.size == 0:
        return b""
    arr = np.packbits(bits_arr, bitorder="big")
    return arr.tobytes()


def _tone_table(freq: float, fs: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=np.float32) / float(fs)
    ang = 2.0 * math.pi * float(freq) * t
    return np.sin(ang), np.cos(ang)


def afsk_modulate(
    bits: Sequence[int],
    *,
    fs: int = 48000,
    baud: int = 1200,
    f_mark: float = 1200.0,
    f_space: float = 2200.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    bits_arr = np.array(bits, dtype=np.uint8, copy=False)
    if bits_arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    n = int(round(float(fs) / float(baud)))
    total = int(bits_arr.size) * n
    out = np.zeros(total, dtype=np.float32)
    phase = 0.0

    for i, bit in enumerate(bits_arr):
        freq = float(f_mark if int(bit) else f_space)
        omega = 2.0 * math.pi * freq / float(fs)
        idx0 = i * n
        idx1 = idx0 + n
        t = np.arange(n, dtype=np.float32)
        out[idx0:idx1] = amplitude * np.sin(phase + omega * t)
        phase = (phase + omega * n) % (2.0 * math.pi)

    return out


def afsk_demodulate(
    samples: np.ndarray,
    *,
    fs: int = 48000,
    baud: int = 1200,
    f_mark: float = 1200.0,
    f_space: float = 2200.0,
) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = int(round(float(fs) / float(baud)))
    if x.size < n:
        return np.zeros(0, dtype=np.uint8)

    total_bits = x.size // n
    x = x[: total_bits * n]
    sin_mark, cos_mark = _tone_table(f_mark, fs, n)
    sin_space, cos_space = _tone_table(f_space, fs, n)

    bits = np.zeros(total_bits, dtype=np.uint8)
    for i in range(total_bits):
        seg = x[i * n:(i + 1) * n]
        sm = float(np.dot(seg, sin_mark))
        cm = float(np.dot(seg, cos_mark))
        ss = float(np.dot(seg, sin_space))
        cs = float(np.dot(seg, cos_space))
        em = sm * sm + cm * cm
        es = ss * ss + cs * cs
        bits[i] = 1 if em > es else 0

    return bits


@dataclass
class AFSKModem:
    fs: int = 48000
    baud: int = 1200
    f_mark: float = 1200.0
    f_space: float = 2200.0
    amplitude: float = 0.8
    preamble_len: int = 16
    decoder: FrameDecoder = field(default_factory=FrameDecoder)

    def encode(self, payload: bytes, *, flags: int = 0) -> np.ndarray:
        frame = build_frame(payload, flags=flags, preamble_len=self.preamble_len)
        bits = bytes_to_bits(frame)
        return afsk_modulate(
            bits,
            fs=self.fs,
            baud=self.baud,
            f_mark=self.f_mark,
            f_space=self.f_space,
            amplitude=self.amplitude,
        )

    def decode(self, samples: np.ndarray) -> List[bytes]:
        bits = afsk_demodulate(
            samples,
            fs=self.fs,
            baud=self.baud,
            f_mark=self.f_mark,
            f_space=self.f_space,
        )
        data = bits_to_bytes(bits)
        return self.decoder.push(data)

    def modulate_bytes(self, data: bytes) -> np.ndarray:
        bits = bytes_to_bits(data)
        return afsk_modulate(
            bits,
            fs=self.fs,
            baud=self.baud,
            f_mark=self.f_mark,
            f_space=self.f_space,
            amplitude=self.amplitude,
        )

    def demodulate_bytes(self, samples: np.ndarray) -> bytes:
        bits = afsk_demodulate(
            samples,
            fs=self.fs,
            baud=self.baud,
            f_mark=self.f_mark,
            f_space=self.f_space,
        )
        return bits_to_bytes(bits)
