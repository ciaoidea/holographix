"""
holo.models.coarse

Pluggable coarse models for v3 codecs (image/audio).
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


try:
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    _BICUBIC = Image.BICUBIC


class CoarseModel:
    name: str = "coarse"
    kind: str = "generic"

    def encode(self, signal: np.ndarray, **params) -> bytes:
        raise NotImplementedError

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        raise NotImplementedError


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

    coords = np.linspace(0.0, base.shape[0] - 1.0, int(block))
    tmp = np.empty((int(block), base.shape[1]), dtype=np.float32)
    for j in range(base.shape[1]):
        tmp[:, j] = np.interp(coords, np.arange(base.shape[0]), base[:, j])
    out = np.empty((int(block), int(block)), dtype=np.float32)
    for i in range(int(block)):
        out[i, :] = np.interp(coords, np.arange(base.shape[1]), tmp[i, :])
    return np.clip(np.round(out), 1.0, 32767.0).astype(np.float32)


def _stft_frame_count(n_samples: int, n_fft: int, hop: int) -> tuple[int, int]:
    if n_samples <= 0:
        return 0, 0
    frames = max(1, int(math.ceil((n_samples - n_fft) / float(hop))) + 1)
    padded_len = n_fft + (frames - 1) * hop
    pad = max(0, padded_len - n_samples)
    return frames, pad


def _stft_1d(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray) -> tuple[np.ndarray, int]:
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


class ImageDownsample(CoarseModel):
    name = "downsample"
    kind = "image"

    def encode(self, signal: np.ndarray, **params) -> bytes:
        max_side = int(params.get("coarse_max_side", 16))
        img = Image.fromarray(np.asarray(signal, dtype=np.uint8), mode="RGB")
        h, w = img.size[1], img.size[0]
        max_dim = max(h, w)
        scale = min(1.0, float(max_side) / float(max_dim))
        cw = max(1, int(round(w * scale)))
        ch = max(1, int(round(h * scale)))
        coarse_img = img.resize((cw, ch), resample=_BICUBIC)
        buf = BytesIO()
        coarse_img.save(buf, format="PNG")
        return buf.getvalue()

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        h, w, _ = target_shape
        with Image.open(BytesIO(payload)) as cim:
            cim = cim.convert("RGB")
            coarse_up = cim.resize((int(w), int(h)), resample=_BICUBIC)
        return np.asarray(coarse_up, dtype=np.int16)


class ImageLowFreq(CoarseModel):
    name = "latent_lowfreq"
    kind = "image"
    _HDR = struct.Struct(">B")

    def encode(self, signal: np.ndarray, **params) -> bytes:
        block = int(params.get("block_size", 8))
        quality = int(params.get("quality", 50))
        lf_keep = int(params.get("lf_keep", 6))
        lf_keep = max(1, min(lf_keep, block * block))

        img = np.asarray(signal, dtype=np.float32)
        h, w, c = img.shape
        pad_h = (block - (h % block)) % block
        pad_w = (block - (w % block)) % block
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        T = _get_dct_matrix(block)
        quant = _scaled_quant_table(block, quality)
        zigzag = _zigzag_indices(block)
        zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]
        blocks_h = img.shape[0] // block
        blocks_w = img.shape[1] // block

        coeff_parts: list[np.ndarray] = []
        for ch_idx in range(c):
            plane = img[:, :, ch_idx]
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
                    coeff_parts.append(q_clip[zz_r, zz_c][:lf_keep].astype(np.int16, copy=False))

        coeffs = np.concatenate(coeff_parts, axis=0) if coeff_parts else np.zeros(0, dtype=np.int16)
        payload = zlib.compress(coeffs.astype("<i2", copy=False).tobytes(), level=9)
        return self._HDR.pack(int(lf_keep)) + payload

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        block = int(params.get("block_size", 8))
        quality = int(params.get("quality", 50))
        if len(payload) < self._HDR.size:
            raise ValueError("latent_lowfreq payload too small")
        lf_keep = int(self._HDR.unpack_from(payload, 0)[0])
        lf_keep = max(1, min(lf_keep, block * block))
        comp = payload[self._HDR.size:]
        coeff_bytes = zlib.decompress(comp)
        coeffs = np.frombuffer(coeff_bytes, dtype="<i2").astype(np.int16, copy=False)

        h, w, c = target_shape
        pad_h = (block - (h % block)) % block
        pad_w = (block - (w % block)) % block
        blocks_h = (h + pad_h) // block
        blocks_w = (w + pad_w) // block
        blocks_per_channel = blocks_h * blocks_w
        expected = blocks_per_channel * int(c) * int(lf_keep)
        if coeffs.size < expected:
            raise ValueError("latent_lowfreq payload underflow")
        coeffs = coeffs[:expected].reshape(int(c), blocks_per_channel, int(lf_keep))

        T = _get_dct_matrix(block)
        quant = _scaled_quant_table(block, quality)
        zigzag = _zigzag_indices(block)
        zz_r, zz_c = zigzag[:, 0], zigzag[:, 1]

        out = np.zeros((blocks_h * block, blocks_w * block, int(c)), dtype=np.float32)
        for ch_idx in range(int(c)):
            ch_blocks = coeffs[ch_idx]
            for bi in range(blocks_per_channel):
                by = bi // blocks_w
                bx = bi - by * blocks_w
                coeff_block = np.zeros((block, block), dtype=np.float32)
                coeff_block[zz_r[:lf_keep], zz_c[:lf_keep]] = ch_blocks[bi].astype(np.float32)
                coeff_block *= quant
                spatial = T.T @ coeff_block @ T
                y0 = by * block
                x0 = bx * block
                out[y0:y0 + block, x0:x0 + block, ch_idx] = spatial

        out = out[:h, :w, :]
        return np.clip(np.round(out), 0.0, 255.0).astype(np.int16)


class AudioDownsample(CoarseModel):
    name = "downsample"
    kind = "audio"

    def encode(self, signal: np.ndarray, **params) -> bytes:
        coarse_len = int(params.get("coarse_len", 2048))
        audio = np.asarray(signal, dtype=np.int16)
        n_frames, ch = audio.shape
        coarse_len = int(min(max(2, coarse_len), n_frames))
        idx = np.linspace(0, n_frames - 1, coarse_len, dtype=np.int64)
        coarse = audio[idx]
        return zlib.compress(coarse.astype("<i2", copy=False).tobytes(), level=9)

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        n_frames, ch = target_shape
        coarse_len = int(params.get("coarse_len", n_frames))
        coarse_bytes = zlib.decompress(payload)
        coarse = np.frombuffer(coarse_bytes, dtype="<i2").astype(np.int16, copy=False)
        coarse = coarse.reshape(int(coarse_len), int(ch))
        t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
        k0 = np.floor(t).astype(np.int64)
        k1 = np.clip(k0 + 1, 0, coarse_len - 1)
        alpha = (t - k0).astype(np.float64)[:, None]
        coarse_f = coarse.astype(np.float64)
        coarse_up = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
        return np.round(coarse_up).astype(np.int16)


class AudioLowFreq(CoarseModel):
    name = "latent_lowfreq"
    kind = "audio"
    _HDR = struct.Struct(">H")

    def encode(self, signal: np.ndarray, **params) -> bytes:
        n_fft = int(params.get("n_fft", 512))
        hop = int(params.get("hop", n_fft // 2))
        quality = int(params.get("quality", 50))
        lf_bins = int(params.get("lf_bins", 16))
        audio = np.asarray(signal, dtype=np.int16)
        n_frames, ch = audio.shape

        hop = max(1, hop)
        window = np.sqrt(np.hanning(n_fft)).astype(np.float32)
        bins = n_fft // 2 + 1
        lf_bins = max(1, min(lf_bins, bins))
        steps = _audio_quant_steps(bins, quality, n_fft)

        parts: list[np.ndarray] = []
        for ch_idx in range(ch):
            spec, _ = _stft_1d(audio[:, ch_idx].astype(np.float32), n_fft, hop, window)
            spec_scaled = spec / float(n_fft)
            q_re = np.rint(np.real(spec_scaled[:, :lf_bins]) / steps[:lf_bins])
            q_im = np.rint(np.imag(spec_scaled[:, :lf_bins]) / steps[:lf_bins])
            q_re = np.clip(q_re, -32768.0, 32767.0).astype(np.int16)
            q_im = np.clip(q_im, -32768.0, 32767.0).astype(np.int16)
            interleaved = np.empty(q_re.size * 2, dtype=np.int16)
            interleaved[0::2] = q_re.reshape(-1)
            interleaved[1::2] = q_im.reshape(-1)
            parts.append(interleaved)

        coeffs = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.int16)
        payload = zlib.compress(coeffs.astype("<i2", copy=False).tobytes(), level=9)
        return self._HDR.pack(int(lf_bins)) + payload

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        n_frames, ch = target_shape
        n_fft = int(params.get("n_fft", 512))
        hop = int(params.get("hop", n_fft // 2))
        quality = int(params.get("quality", 50))
        hop = max(1, hop)
        if len(payload) < self._HDR.size:
            raise ValueError("latent_lowfreq payload too small")
        lf_bins = int(self._HDR.unpack_from(payload, 0)[0])
        comp = payload[self._HDR.size:]
        coeff_bytes = zlib.decompress(comp)
        coeffs = np.frombuffer(coeff_bytes, dtype="<i2").astype(np.int16, copy=False)

        frames, _ = _stft_frame_count(int(n_frames), int(n_fft), int(hop))
        bins = n_fft // 2 + 1
        lf_bins = max(1, min(lf_bins, bins))
        expected = int(ch) * int(frames) * int(lf_bins) * 2
        if coeffs.size < expected:
            raise ValueError("latent_lowfreq payload underflow")
        coeffs = coeffs[:expected].reshape(int(ch), int(frames), int(lf_bins) * 2)

        window = np.sqrt(np.hanning(n_fft)).astype(np.float32)
        steps = _audio_quant_steps(bins, quality, n_fft)
        out = np.zeros((int(n_frames), int(ch)), dtype=np.float64)

        for ch_idx in range(int(ch)):
            q_re = coeffs[ch_idx, :, 0::2].astype(np.float32)
            q_im = coeffs[ch_idx, :, 1::2].astype(np.float32)
            spec = np.zeros((int(frames), int(bins)), dtype=np.complex64)
            spec[:, :lf_bins] = (q_re + 1j * q_im) * steps[None, :lf_bins]
            spec *= float(n_fft)
            out[:, ch_idx] = _istft_1d(spec, int(n_fft), int(hop), window, int(n_frames))

        return np.round(out).astype(np.int16)


@dataclass
class _AeWeights:
    enc_w: np.ndarray
    enc_b: np.ndarray
    dec_w: np.ndarray
    dec_b: np.ndarray
    input_shape: tuple[int, ...]


def _load_ae_weights(path: Path) -> Optional[_AeWeights]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        enc_w = data["enc_w"]
        enc_b = data["enc_b"]
        dec_w = data["dec_w"]
        dec_b = data["dec_b"]
        input_shape = tuple(int(x) for x in data["input_shape"].tolist())
    except Exception:
        return None
    return _AeWeights(enc_w=enc_w, enc_b=enc_b, dec_w=dec_w, dec_b=dec_b, input_shape=input_shape)


class AeLatentImage(CoarseModel):
    name = "ae_latent"
    kind = "image"
    _HDR = struct.Struct(">II")

    def __init__(self) -> None:
        self._weights_path = Path(__file__).with_name("ae_latent_image.npz")
        self._weights: Optional[_AeWeights] = None

    def _weights_or_none(self) -> Optional[_AeWeights]:
        if self._weights is None:
            self._weights = _load_ae_weights(self._weights_path)
        return self._weights

    def encode(self, signal: np.ndarray, **params) -> bytes:
        weights = self._weights_or_none()
        png_bytes = ImageDownsample().encode(signal, **params)
        latent_bytes = b""
        if weights is not None:
            img = np.asarray(signal, dtype=np.float32)
            h0, w0, c0 = weights.input_shape
            if img.shape[2] == c0:
                img_small = Image.fromarray(np.asarray(signal, dtype=np.uint8), mode="RGB").resize((w0, h0), resample=_BICUBIC)
                img_small_f = np.asarray(img_small, dtype=np.float32) / 255.0
                x = img_small_f.reshape(-1)
                z = np.tanh(weights.enc_w @ x + weights.enc_b)
                latent = z.astype(np.float32)
                latent_bytes = zlib.compress(latent.tobytes(), level=9)
        return self._HDR.pack(int(len(png_bytes)), int(len(latent_bytes))) + png_bytes + latent_bytes

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        weights = self._weights_or_none()
        if len(payload) < self._HDR.size:
            raise ValueError("ae_latent payload too small")
        png_len, latent_len = self._HDR.unpack_from(payload, 0)
        off = self._HDR.size
        if off + png_len + latent_len > len(payload):
            raise ValueError("ae_latent payload truncated")
        if weights is None or latent_len <= 0:
            return ImageDownsample().decode(payload[off: off + png_len], target_shape, **params)
        latent_bytes = payload[off + png_len: off + png_len + latent_len]
        try:
            latent = np.frombuffer(zlib.decompress(latent_bytes), dtype=np.float32)
        except Exception:
            return ImageDownsample().decode(payload[off: off + png_len], target_shape, **params)
        y = weights.dec_w @ latent + weights.dec_b
        h0, w0, c0 = weights.input_shape
        recon = y.reshape(h0, w0, c0)
        recon = np.clip(recon * 255.0, 0.0, 255.0).astype(np.uint8)
        img = Image.fromarray(recon, mode="RGB").resize((int(target_shape[1]), int(target_shape[0])), resample=_BICUBIC)
        return np.asarray(img, dtype=np.int16)


class AeLatentAudio(CoarseModel):
    name = "ae_latent"
    kind = "audio"
    _HDR = struct.Struct(">II")

    def __init__(self) -> None:
        self._weights_path = Path(__file__).with_name("ae_latent_audio.npz")
        self._weights: Optional[_AeWeights] = None

    def _weights_or_none(self) -> Optional[_AeWeights]:
        if self._weights is None:
            self._weights = _load_ae_weights(self._weights_path)
        return self._weights

    def encode(self, signal: np.ndarray, **params) -> bytes:
        weights = self._weights_or_none()
        coarse_bytes = AudioDownsample().encode(signal, **params)
        latent_bytes = b""
        if weights is not None:
            audio = np.asarray(signal, dtype=np.float32)
            n_frames, ch = audio.shape
            if ch == 1:
                target_len = int(weights.input_shape[0])
                idx = np.linspace(0, n_frames - 1, target_len, dtype=np.int64)
                mono = audio[idx, 0] / 32768.0
                z = np.tanh(weights.enc_w @ mono + weights.enc_b)
                latent = z.astype(np.float32)
                latent_bytes = zlib.compress(latent.tobytes(), level=9)
        return self._HDR.pack(int(len(coarse_bytes)), int(len(latent_bytes))) + coarse_bytes + latent_bytes

    def decode(self, payload: bytes, target_shape, **params) -> np.ndarray:
        weights = self._weights_or_none()
        if len(payload) < self._HDR.size:
            raise ValueError("ae_latent payload too small")
        coarse_len, latent_len = self._HDR.unpack_from(payload, 0)
        off = self._HDR.size
        if off + coarse_len + latent_len > len(payload):
            raise ValueError("ae_latent payload truncated")
        if weights is None or latent_len <= 0:
            return AudioDownsample().decode(payload[off: off + coarse_len], target_shape, **params)
        latent_bytes = payload[off + coarse_len: off + coarse_len + latent_len]
        try:
            latent = np.frombuffer(zlib.decompress(latent_bytes), dtype=np.float32)
        except Exception:
            return AudioDownsample().decode(payload[off: off + coarse_len], target_shape, **params)
        y = weights.dec_w @ latent + weights.dec_b
        n_frames, ch = target_shape
        recon = np.clip(y, -1.0, 1.0) * 32768.0
        x = np.interp(
            np.linspace(0, recon.size - 1, int(n_frames)),
            np.arange(recon.size),
            recon,
        )
        return np.round(x).astype(np.int16).reshape(int(n_frames), 1)


_IMAGE_MODELS = {
    "downsample": ImageDownsample(),
    "latent_lowfreq": ImageLowFreq(),
    "ae_latent": AeLatentImage(),
}
_AUDIO_MODELS = {
    "downsample": AudioDownsample(),
    "latent_lowfreq": AudioLowFreq(),
    "ae_latent": AeLatentAudio(),
}


def get_coarse_model(name: str, *, kind: str) -> CoarseModel:
    name_norm = (name or "downsample").strip().lower()
    if kind == "audio":
        model = _AUDIO_MODELS.get(name_norm)
        if model is None:
            return _AUDIO_MODELS["downsample"]
        return model
    model = _IMAGE_MODELS.get(name_norm)
    if model is None:
        return _IMAGE_MODELS["downsample"]
    return model
