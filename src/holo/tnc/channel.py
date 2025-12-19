"""
holo.tnc.channel

Simple channel effects for TNC simulation.
"""

from __future__ import annotations

import numpy as np


def awgn(samples: np.ndarray, snr_db: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x
    rng = rng or np.random.default_rng()
    power = float(np.mean(x * x))
    if power <= 0.0:
        return x
    snr = 10.0 ** (float(snr_db) / 10.0)
    noise_power = power / snr
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return x + noise


def dropout(
    samples: np.ndarray,
    *,
    drop_prob: float = 0.0,
    drop_len_sec: float = 0.02,
    fs: int = 48000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    x = np.array(samples, dtype=np.float32, copy=True)
    if x.size == 0 or drop_prob <= 0.0:
        return x
    rng = rng or np.random.default_rng()
    if rng.random() >= float(drop_prob):
        return x
    n = int(max(1, round(float(drop_len_sec) * float(fs))))
    if n >= x.size:
        x[:] = 0.0
        return x
    start = int(rng.integers(0, max(1, x.size - n)))
    x[start:start + n] = 0.0
    return x
