"""
holo.field

Local field helper for a single content_id.

Tracks chunk coverage, decodes the best current percept, and can heal
back into a fresh holographic population.
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
from typing import Optional

import numpy as np
from PIL import Image

from .container import unpack_object_from_holo_dir
from .codec import (
    MAGIC_AUD,
    MAGIC_IMG,
    VERSION_AUD_OLO,
    VERSION_IMG_OLO,
    _confidence_frames_audio_v3,
    _confidence_map_image_v3,
    _load_audio_field_v3,
    _load_image_field_v3,
    _render_audio_field_v3,
    _render_image_field_v3,
    _write_audio_field_v3,
    _write_image_field_v3,
    _write_wav_int16,
    decode_audio_holo_dir,
    decode_audio_holo_dir_meta,
    decode_image_holo_dir,
    decode_image_holo_dir_meta,
    detect_mode_from_chunk_dir,
    encode_audio_holo_dir,
    encode_image_holo_dir,
    save_image_rgb_u8,
)

try:
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # pragma: no cover
    _BICUBIC = Image.BICUBIC


class Field:
    """
    Manage a local holographic field for one content_id.
    """

    def __init__(self, content_id: str, chunk_dir: str) -> None:
        self.content_id = content_id
        self.chunk_dir = chunk_dir

    def coverage(self) -> dict:
        """
        Report how many chunks are present versus expected.
        """
        present = len(self._chunk_files())
        meta = self._detect_layout()
        total = meta.get("block_count", present)
        return {
            "content_id": self.content_id,
            "present_blocks": present,
            "total_blocks": total,
            "layout": meta.get("layout", "unknown"),
        }

    def best_decode_image(self, *, max_chunks: Optional[int] = None, object_index: int = 0) -> str:
        """
        Decode the best available image to a derived output path and return it.
        """
        out_path = self._default_output(ext=".png")
        if self._is_container():
            unpack_object_from_holo_dir(self.chunk_dir, object_index, out_path, max_chunks=max_chunks)
            return out_path

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        if mode != "image":
            raise ValueError("Field is not image-capable")
        decode_image_holo_dir(self.chunk_dir, out_path, max_chunks=max_chunks)
        return out_path

    def best_decode_audio(self, *, max_chunks: Optional[int] = None, object_index: int = 0) -> str:
        """
        Decode the best available audio to a derived output path and return it.
        """
        out_path = self._default_output(ext=".wav")
        if self._is_container():
            unpack_object_from_holo_dir(self.chunk_dir, object_index, out_path, max_chunks=max_chunks)
            return out_path

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        if mode != "audio":
            raise ValueError("Field is not audio-capable")
        decode_audio_holo_dir(self.chunk_dir, out_path, max_chunks=max_chunks)
        return out_path

    def best_decode_image_meta(
        self,
        *,
        max_chunks: Optional[int] = None,
        object_index: int = 0,
        prefer_gain: bool = False,
        use_recovery: Optional[bool] = None,
    ) -> tuple[str, str]:
        """
        Decode the best available image plus confidence map and return both paths.
        """
        out_path = self._default_output(ext=".png")
        conf_path = self._confidence_output(out_path, kind="image")
        if self._is_container():
            raise NotImplementedError("Meta decode for packed fields is not yet supported")

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        if mode != "image":
            raise ValueError("Field is not image-capable")
        recon, conf = decode_image_holo_dir_meta(
            self.chunk_dir,
            max_chunks=max_chunks,
            prefer_gain=prefer_gain,
            use_recovery=use_recovery,
        )
        save_image_rgb_u8(recon, out_path)
        conf_u8 = np.clip(conf * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(conf_u8, mode="L").save(conf_path)
        return out_path, conf_path

    def best_decode_audio_meta(
        self,
        *,
        max_chunks: Optional[int] = None,
        object_index: int = 0,
        prefer_gain: bool = False,
        use_recovery: Optional[bool] = None,
    ) -> tuple[str, str]:
        """
        Decode the best available audio plus confidence curve and return both paths.
        """
        out_path = self._default_output(ext=".wav")
        conf_path = self._confidence_output(out_path, kind="audio")
        if self._is_container():
            raise NotImplementedError("Meta decode for packed fields is not yet supported")

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        if mode != "audio":
            raise ValueError("Field is not audio-capable")
        audio, sr, conf = decode_audio_holo_dir_meta(
            self.chunk_dir,
            max_chunks=max_chunks,
            prefer_gain=prefer_gain,
            use_recovery=use_recovery,
            return_sr=True,
        )
        _write_wav_int16(out_path, audio, int(sr))
        np.save(conf_path, conf.astype(np.float32))
        return out_path, conf_path

    def heal_to(
        self,
        out_dir: str,
        *,
        target_chunk_kb: int = 32,
        max_chunks: Optional[int] = None,
        honest: bool = False,
        prefer_gain: bool = False,
        use_recovery: Optional[bool] = None,
    ) -> str:
        """
        Re-encode the best current percept into a new holographic population.
        """
        return self.heal_once(
            out_dir,
            target_chunk_kb=target_chunk_kb,
            max_chunks=max_chunks,
            honest=honest,
            prefer_gain=prefer_gain,
            use_recovery=use_recovery,
        )

    def heal_once(
        self,
        out_dir: str,
        *,
        target_chunk_kb: int = 32,
        max_chunks: Optional[int] = None,
        honest: bool = False,
        prefer_gain: bool = False,
        use_recovery: Optional[bool] = None,
        return_metrics: bool = False,
    ):
        """
        Perform one healing step and optionally return a metrics dict.
        """
        if self._is_container():
            raise NotImplementedError("Healing packed fields is not yet supported")

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        metrics: dict[str, float] = {}
        if mode == "audio":
            if self._chunk_version() == VERSION_AUD_OLO:
                state = _load_audio_field_v3(
                    self.chunk_dir,
                    max_chunks=max_chunks,
                    prefer_gain=prefer_gain,
                    use_recovery=use_recovery,
                    return_mask=True,
                )
                coeff = state.coeff_vec.astype(np.float32, copy=True)
                total_expected = int(state.ch) * int(state.frames) * int(state.bins) * 2
                coeff_use = coeff[:total_expected].reshape(int(state.ch), int(state.frames), int(state.bins) * 2)
                if honest:
                    conf_frames = _confidence_frames_audio_v3(state)
                    metrics["confidence_mean"] = float(np.mean(conf_frames))
                    coeff_use *= conf_frames[None, :, None]
                coeff = coeff_use.reshape(-1)
                state.coeff_vec = coeff
                _write_audio_field_v3(state, out_dir, target_chunk_kb=target_chunk_kb)
            else:
                audio, sr, conf = decode_audio_holo_dir_meta(
                    self.chunk_dir,
                    max_chunks=max_chunks,
                    prefer_gain=prefer_gain,
                    use_recovery=use_recovery,
                    return_sr=True,
                )
                if honest:
                    audio = self._attenuate_audio(audio, conf)
                    metrics["confidence_mean"] = float(np.mean(conf))
                tmp = self._write_temp_wav(audio, int(sr))
                try:
                    encode_audio_holo_dir(tmp, out_dir, target_chunk_kb=target_chunk_kb)
                finally:
                    self._cleanup_temp(tmp)
        else:
            if self._chunk_version() == VERSION_IMG_OLO:
                state = _load_image_field_v3(
                    self.chunk_dir,
                    max_chunks=max_chunks,
                    prefer_gain=prefer_gain,
                    use_recovery=use_recovery,
                    return_mask=True,
                )
                coeff = state.coeff_vec.astype(np.float32, copy=True)
                blocks_per_channel = int(state.blocks_h) * int(state.blocks_w)
                coeff_per_block = int(state.block_size) * int(state.block_size)
                total_expected = blocks_per_channel * coeff_per_block * int(state.c)
                coeff_use = coeff[:total_expected].reshape(int(state.c), blocks_per_channel, coeff_per_block)
                if honest:
                    conf_blocks = _confidence_map_image_v3(state)
                    metrics["confidence_mean"] = float(np.mean(conf_blocks))
                    conf_flat = conf_blocks.reshape(blocks_per_channel).astype(np.float32, copy=False)
                    coeff_use *= conf_flat[None, :, None]
                coeff = coeff_use.reshape(-1)
                state.coeff_vec = coeff
                _write_image_field_v3(state, out_dir, target_chunk_kb=target_chunk_kb)
            else:
                recon, conf = decode_image_holo_dir_meta(
                    self.chunk_dir,
                    max_chunks=max_chunks,
                    prefer_gain=prefer_gain,
                    use_recovery=use_recovery,
                )
                if honest:
                    recon = self._attenuate_image(recon, conf)
                    metrics["confidence_mean"] = float(np.mean(conf))
                tmp = self._write_temp_image(recon)
                try:
                    encode_image_holo_dir(tmp, out_dir, target_chunk_kb=target_chunk_kb)
                finally:
                    self._cleanup_temp(tmp)
        metrics["honest"] = float(bool(honest))
        if return_metrics:
            return out_dir, metrics
        return out_dir

    def heal_fixed_point(
        self,
        out_dir: str,
        *,
        max_iters: int = 4,
        tol: float = 1e-3,
        metric: str = "mse",
        target_chunk_kb: int = 32,
        max_chunks: Optional[int] = None,
        honest: bool = False,
        prefer_gain: bool = False,
        use_recovery: Optional[bool] = None,
    ) -> dict:
        """
        Apply heal_once iteratively until convergence.
        """
        if self._is_container():
            raise NotImplementedError("Healing packed fields is not yet supported")

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        metric = str(metric).lower()
        max_iters = max(1, int(max_iters))
        tol = float(tol)
        use_v3 = False
        if mode == "audio" and self._chunk_version() == VERSION_AUD_OLO:
            use_v3 = True
        if mode == "image" and self._chunk_version() == VERSION_IMG_OLO:
            use_v3 = True

        deltas: list[float] = []
        drift: list[float] = []
        dirs: list[str] = []
        prev_recon = None
        first_recon = None
        current_dir = self.chunk_dir

        for i in range(max_iters):
            if mode == "audio":
                if use_v3:
                    state = _load_audio_field_v3(
                        current_dir,
                        max_chunks=max_chunks,
                        prefer_gain=prefer_gain,
                        use_recovery=use_recovery,
                        return_mask=True,
                    )
                    coeff = state.coeff_vec.astype(np.float32, copy=True)
                    total_expected = int(state.ch) * int(state.frames) * int(state.bins) * 2
                    coeff_use = coeff[:total_expected].reshape(int(state.ch), int(state.frames), int(state.bins) * 2)
                    if honest:
                        conf_frames = _confidence_frames_audio_v3(state)
                        coeff_use *= conf_frames[None, :, None]
                    coeff = coeff_use.reshape(-1)
                    recon = _render_audio_field_v3(state, coeff_vec=coeff).astype(np.float32)
                    max_val = 32767.0
                else:
                    audio, sr, conf = decode_audio_holo_dir_meta(
                        current_dir,
                        max_chunks=max_chunks,
                        prefer_gain=prefer_gain,
                        use_recovery=use_recovery,
                        return_sr=True,
                    )
                    if honest:
                        audio_use = self._attenuate_audio(audio, conf)
                    else:
                        audio_use = audio
                    recon = audio_use.astype(np.float32)
                    max_val = 32767.0
            else:
                if use_v3:
                    state = _load_image_field_v3(
                        current_dir,
                        max_chunks=max_chunks,
                        prefer_gain=prefer_gain,
                        use_recovery=use_recovery,
                        return_mask=True,
                    )
                    coeff = state.coeff_vec.astype(np.float32, copy=True)
                    blocks_per_channel = int(state.blocks_h) * int(state.blocks_w)
                    coeff_per_block = int(state.block_size) * int(state.block_size)
                    total_expected = blocks_per_channel * coeff_per_block * int(state.c)
                    coeff_use = coeff[:total_expected].reshape(int(state.c), blocks_per_channel, coeff_per_block)
                    if honest:
                        conf_blocks = _confidence_map_image_v3(state)
                        conf_flat = conf_blocks.reshape(blocks_per_channel).astype(np.float32, copy=False)
                        coeff_use *= conf_flat[None, :, None]
                    coeff = coeff_use.reshape(-1)
                    recon = _render_image_field_v3(state, coeff_vec=coeff).astype(np.float32)
                    max_val = 255.0
                else:
                    recon_img, conf = decode_image_holo_dir_meta(
                        current_dir,
                        max_chunks=max_chunks,
                        prefer_gain=prefer_gain,
                        use_recovery=use_recovery,
                    )
                    if honest:
                        recon_use = self._attenuate_image(recon_img, conf)
                    else:
                        recon_use = recon_img
                    recon = recon_use.astype(np.float32)
                    max_val = 255.0

            if first_recon is None:
                first_recon = recon
            if prev_recon is not None:
                delta = self._metric_value(prev_recon, recon, metric, max_val)
                deltas.append(delta)
                drift.append(self._metric_value(first_recon, recon, metric, max_val))
                if len(deltas) >= 2:
                    if metric == "psnr":
                        if deltas[-1] < deltas[-2] - 1e-9:
                            break
                        improvement = deltas[-1] - deltas[-2]
                    else:
                        if deltas[-1] > deltas[-2] + 1e-9:
                            break
                        improvement = deltas[-2] - deltas[-1]
                    if improvement < tol:
                        break

            iter_dir = out_dir if i == 0 else f"{out_dir}_iter{i + 1}"
            dirs.append(iter_dir)
            if mode == "audio":
                if use_v3:
                    state.coeff_vec = coeff
                    _write_audio_field_v3(state, iter_dir, target_chunk_kb=target_chunk_kb)
                else:
                    tmp = self._write_temp_wav(audio_use, int(sr))
                    try:
                        encode_audio_holo_dir(tmp, iter_dir, target_chunk_kb=target_chunk_kb)
                    finally:
                        self._cleanup_temp(tmp)
            else:
                if use_v3:
                    state.coeff_vec = coeff
                    _write_image_field_v3(state, iter_dir, target_chunk_kb=target_chunk_kb)
                else:
                    tmp = self._write_temp_image(recon_use.astype(np.uint8))
                    try:
                        encode_image_holo_dir(tmp, iter_dir, target_chunk_kb=target_chunk_kb)
                    finally:
                        self._cleanup_temp(tmp)

            prev_recon = recon
            current_dir = iter_dir

        return {
            "iterations": len(dirs),
            "deltas": deltas,
            "drift": drift,
            "metric": metric,
            "dirs": dirs,
            "out_dir": dirs[-1] if dirs else out_dir,
        }

    def _confidence_output(self, out_path: str, *, kind: str) -> str:
        root, _ext = os.path.splitext(out_path)
        if kind == "audio":
            return root + "_confidence.npy"
        return root + "_confidence.png"

    @staticmethod
    def _metric_value(a: np.ndarray, b: np.ndarray, metric: str, max_val: float) -> float:
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        mse = float(np.mean((a_f - b_f) ** 2))
        if metric == "psnr":
            if mse == 0.0:
                return float("inf")
            return 10.0 * math.log10((float(max_val) ** 2) / mse)
        if metric == "l2":
            return float(math.sqrt(mse))
        return mse

    @staticmethod
    def _write_temp_image(arr: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        save_image_rgb_u8(arr, path)
        return path

    @staticmethod
    def _write_temp_wav(audio: np.ndarray, sr: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _write_wav_int16(path, audio, int(sr))
        return path

    @staticmethod
    def _cleanup_temp(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _attenuate_image(img: np.ndarray, conf: np.ndarray, *, coarse_max_side: int = 16) -> np.ndarray:
        h, w, _ = img.shape
        max_side = max(h, w)
        scale = min(1.0, float(coarse_max_side) / float(max_side))
        cw = max(1, int(round(w * scale)))
        ch = max(1, int(round(h * scale)))
        img_pil = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB")
        coarse = img_pil.resize((cw, ch), resample=_BICUBIC).resize((w, h), resample=_BICUBIC)
        coarse_arr = np.asarray(coarse, dtype=np.float32)

        blocks_h, blocks_w = conf.shape
        block_h = int(math.ceil(float(h) / float(blocks_h))) if blocks_h else h
        block_w = int(math.ceil(float(w) / float(blocks_w))) if blocks_w else w
        conf_up = np.repeat(np.repeat(conf, block_h, axis=0), block_w, axis=1)
        conf_up = conf_up[:h, :w]
        conf_up = np.clip(conf_up, 0.0, 1.0)

        resid = img.astype(np.float32) - coarse_arr
        healed = coarse_arr + resid * conf_up[:, :, None]
        return np.clip(np.round(healed), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _attenuate_audio(audio: np.ndarray, conf: np.ndarray, *, coarse_max_frames: int = 2048) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.int16)
        n_frames, ch = audio.shape
        coarse_len = int(min(max(2, coarse_max_frames), n_frames))
        idx = np.linspace(0, n_frames - 1, coarse_len, dtype=np.int64)
        coarse = audio[idx]
        t = np.linspace(0, coarse_len - 1, n_frames, dtype=np.float64)
        k0 = np.floor(t).astype(np.int64)
        k1 = np.clip(k0 + 1, 0, coarse_len - 1)
        alpha = (t - k0).astype(np.float64)[:, None]
        coarse_f = coarse.astype(np.float64)
        coarse_up = (1.0 - alpha) * coarse_f[k0] + alpha * coarse_f[k1]
        coarse_up = np.round(coarse_up).astype(np.float32)

        conf_curve = np.asarray(conf, dtype=np.float32).reshape(-1)
        if conf_curve.size != n_frames:
            conf_curve = np.interp(
                np.linspace(0.0, float(conf_curve.size - 1), n_frames),
                np.arange(conf_curve.size),
                conf_curve,
            ).astype(np.float32)
        conf_curve = np.clip(conf_curve, 0.0, 1.0)

        resid = audio.astype(np.float32) - coarse_up
        healed = coarse_up + resid * conf_curve[:, None]
        return np.clip(np.round(healed), -32768.0, 32767.0).astype(np.int16)

    # ----------------- internal helpers -----------------

    def _chunk_version(self, chunk_dir: Optional[str] = None) -> Optional[int]:
        use_dir = chunk_dir or self.chunk_dir
        try:
            names = sorted(
                f for f in os.listdir(use_dir)
                if f.startswith("chunk_") and f.endswith(".holo")
            )
        except OSError:
            return None
        if not names:
            return None
        path = os.path.join(use_dir, names[0])
        try:
            with open(path, "rb") as f:
                head = f.read(5)
        except OSError:
            return None
        if len(head) < 5:
            return None
        magic = head[:4]
        if magic not in (MAGIC_IMG, MAGIC_AUD):
            return None
        return int(head[4])

    def _chunk_files(self):
        return sorted(
            f for f in os.listdir(self.chunk_dir)
            if f.startswith("chunk_") and f.endswith(".holo")
        )

    def _is_container(self) -> bool:
        manifest_path = os.path.join(self.chunk_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            return False
        try:
            import json

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, dict) and "objects" in data
        except Exception:
            return False

    def _default_output(self, ext: str) -> str:
        base = self.chunk_dir[:-5] if self.chunk_dir.lower().endswith(".holo") else self.chunk_dir
        return base + "_recon" + ext

    def _detect_layout(self) -> dict:
        if self._is_container():
            return {"layout": "container", "block_count": self._read_manifest_block_count()}

        chunk_files = self._chunk_files()
        if not chunk_files:
            return {"layout": "unknown", "block_count": 0}

        first = os.path.join(self.chunk_dir, chunk_files[0])
        try:
            with open(first, "rb") as f:
                data = f.read(32)
        except OSError:
            return {"layout": "unknown", "block_count": 0}

        magic = data[:4]
        if magic == MAGIC_IMG and len(data) >= 14 + 4:
            block_count = struct.unpack(">I", data[14:18])[0]
            return {"layout": "image", "block_count": block_count}
        if magic == MAGIC_AUD and len(data) >= 16 + 4:
            block_count = struct.unpack(">I", data[16:20])[0]
            return {"layout": "audio", "block_count": block_count}
        return {"layout": "unknown", "block_count": len(chunk_files)}

    def _read_manifest_block_count(self) -> int:
        manifest_path = os.path.join(self.chunk_dir, "manifest.json")
        try:
            import json

            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "objects" in data:
                return int(data.get("block_count", 0))
            return 0
        except Exception:
            return 0
