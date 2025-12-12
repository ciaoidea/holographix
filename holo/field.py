"""
holo.field

Local field helper for a single content_id.

Tracks chunk coverage, decodes the best current percept, and can heal
back into a fresh holographic population.
"""

from __future__ import annotations

import os
import struct
from typing import Optional

from .container import unpack_object_from_holo_dir
from .codec import (
    MAGIC_AUD,
    MAGIC_IMG,
    detect_mode_from_chunk_dir,
    decode_audio_holo_dir,
    decode_image_holo_dir,
    encode_audio_holo_dir,
    encode_image_holo_dir,
)


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

    def heal_to(
        self,
        out_dir: str,
        *,
        target_chunk_kb: int = 32,
        max_chunks: Optional[int] = None,
    ) -> str:
        """
        Re-encode the best current percept into a new holographic population.
        """
        if self._is_container():
            raise NotImplementedError("Healing packed fields is not yet supported")

        mode = detect_mode_from_chunk_dir(self.chunk_dir)
        if mode == "audio":
            tmp = self.best_decode_audio(max_chunks=max_chunks)
            encode_audio_holo_dir(tmp, out_dir, target_chunk_kb=target_chunk_kb)
        else:
            tmp = self.best_decode_image(max_chunks=max_chunks)
            encode_image_holo_dir(tmp, out_dir, target_chunk_kb=target_chunk_kb)
        return out_dir

    # ----------------- internal helpers -----------------

    def _chunk_files(self):
        return sorted(
            f for f in os.listdir(self.chunk_dir)
            if f.startswith("chunk_") and f.endswith(".holo")
        )

    def _is_container(self) -> bool:
        return os.path.isfile(os.path.join(self.chunk_dir, "manifest.json"))

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
            return int(data.get("block_count", 0))
        except Exception:
            return 0
