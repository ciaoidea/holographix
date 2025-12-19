"""
holo.tv

HoloTV helpers for scheduling and receiving multi-frame holographic chunks.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

from ..cortex.store import CortexStore
from ..net.arch import content_id_bytes_from_stream_frame
from ..net.transport import ChunkAssembler, iter_chunk_datagrams

try:  # Optional; only used for convenience helpers.
    from ..field import Field  # type: ignore
except Exception:  # pragma: no cover
    Field = None  # type: ignore

__all__ = [
    "HoloTVFrame",
    "HoloTVChunkRef",
    "HoloTVChunk",
    "HoloTVReceipt",
    "HoloTVWindow",
    "HoloTVReceiver",
    "frame_content_id",
]

_RECOVERY_ID_BASE = 1_000_000


def frame_content_id(stream_id: str, frame_idx: int) -> bytes:
    """
    Deterministic content_id for a stream frame.
    """
    return content_id_bytes_from_stream_frame(str(stream_id), int(frame_idx))


@dataclass(frozen=True)
class HoloTVFrame:
    frame_idx: int
    chunk_dir: str


@dataclass(frozen=True)
class HoloTVChunkRef:
    frame_idx: int
    content_id: bytes
    chunk_id: int
    path: str
    is_recovery: bool = False


@dataclass(frozen=True)
class HoloTVChunk:
    frame_idx: int
    content_id: bytes
    chunk_id: int
    data: bytes
    path: str
    is_recovery: bool = False


@dataclass(frozen=True)
class HoloTVReceipt:
    content_id: bytes
    chunk_id: int
    stored_path: str
    frame_idx: Optional[int] = None


def _chunk_id_from_name(path: str) -> int:
    base = os.path.basename(path)
    if base.startswith("chunk_") and base.endswith(".holo"):
        try:
            return int(base[6:-5])
        except ValueError:
            return 0
    return 0


def _recovery_id_from_name(path: str) -> int:
    base = os.path.basename(path)
    if base.startswith("recovery_") and base.endswith(".holo"):
        try:
            return int(base[9:-5])
        except ValueError:
            return 0
    return 0


def _load_manifest(chunk_dir: str) -> Optional[dict]:
    path = os.path.join(chunk_dir, "manifest.json")
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


def _read_chunk_score(meta_path: str) -> float:
    try:
        with open(meta_path, "r", encoding="ascii") as f:
            raw = f.read().strip()
        return float(raw)
    except Exception:
        return 0.0


def _ordered_chunk_paths(chunk_dir: str, *, prefer_gain: bool) -> List[str]:
    chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.holo")))
    if not prefer_gain or not chunk_paths:
        return chunk_paths

    manifest = _load_manifest(chunk_dir)
    if manifest and isinstance(manifest.get("ordered_chunks"), list):
        ordered = []
        for name in manifest["ordered_chunks"]:
            path = os.path.join(chunk_dir, str(name))
            if os.path.isfile(path):
                ordered.append(path)
        if ordered:
            return ordered

    scored: List[tuple[float, str]] = []
    for path in chunk_paths:
        score = _read_chunk_score(path + ".meta")
        scored.append((float(score), path))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


class HoloTVWindow:
    """
    Schedule chunks across a window of frames.
    """

    def __init__(
        self,
        stream_id: str,
        frames: Sequence[HoloTVFrame],
        *,
        prefer_gain: bool = False,
        include_recovery: bool = False,
    ) -> None:
        self.stream_id = str(stream_id)
        self.frames = list(frames)
        self.prefer_gain = bool(prefer_gain)
        self.include_recovery = bool(include_recovery)
        self._content_ids = {
            int(frame.frame_idx): frame_content_id(self.stream_id, int(frame.frame_idx))
            for frame in self.frames
        }

    @classmethod
    def from_chunk_dirs(
        cls,
        stream_id: str,
        chunk_dirs: Sequence[str],
        *,
        frame_start: int = 0,
        prefer_gain: bool = False,
        include_recovery: bool = False,
    ) -> "HoloTVWindow":
        frames = [HoloTVFrame(frame_start + idx, path) for idx, path in enumerate(chunk_dirs)]
        return cls(
            stream_id,
            frames,
            prefer_gain=prefer_gain,
            include_recovery=include_recovery,
        )

    def content_id_for_frame(self, frame_idx: int) -> bytes:
        return self._content_ids.get(int(frame_idx)) or frame_content_id(self.stream_id, int(frame_idx))

    def _frame_chunk_paths(
        self,
        frame: HoloTVFrame,
        *,
        max_chunks: Optional[int],
    ) -> List[str]:
        paths = _ordered_chunk_paths(frame.chunk_dir, prefer_gain=self.prefer_gain)
        if max_chunks is not None:
            paths = paths[: max(1, int(max_chunks))]
        if self.include_recovery:
            rec_paths = sorted(glob.glob(os.path.join(frame.chunk_dir, "recovery_*.holo")))
            if rec_paths:
                paths = paths + rec_paths
        return paths

    def iter_chunk_refs(
        self,
        *,
        order: str = "round_robin",
        max_chunks_per_frame: Optional[int] = None,
    ) -> Iterator[HoloTVChunkRef]:
        if not self.frames:
            return

        per_frame = [
            self._frame_chunk_paths(frame, max_chunks=max_chunks_per_frame)
            for frame in self.frames
        ]

        if order == "sequential":
            for frame, paths in zip(self.frames, per_frame):
                cid = self.content_id_for_frame(frame.frame_idx)
                for path in paths:
                    base = os.path.basename(path)
                    is_recovery = base.startswith("recovery_")
                    if is_recovery:
                        chunk_id = _RECOVERY_ID_BASE + _recovery_id_from_name(path)
                    else:
                        chunk_id = _chunk_id_from_name(path)
                    yield HoloTVChunkRef(frame.frame_idx, cid, int(chunk_id), path, is_recovery=is_recovery)
            return

        if order != "round_robin":
            raise ValueError("order must be 'round_robin' or 'sequential'")

        idx = 0
        while True:
            emitted = False
            for frame, paths in zip(self.frames, per_frame):
                if idx >= len(paths):
                    continue
                emitted = True
                path = paths[idx]
                base = os.path.basename(path)
                is_recovery = base.startswith("recovery_")
                if is_recovery:
                    chunk_id = _RECOVERY_ID_BASE + _recovery_id_from_name(path)
                else:
                    chunk_id = _chunk_id_from_name(path)
                cid = self.content_id_for_frame(frame.frame_idx)
                yield HoloTVChunkRef(frame.frame_idx, cid, int(chunk_id), path, is_recovery=is_recovery)
            if not emitted:
                break
            idx += 1
        return

    def iter_chunks(
        self,
        *,
        order: str = "round_robin",
        max_chunks_per_frame: Optional[int] = None,
    ) -> Iterator[HoloTVChunk]:
        for ref in self.iter_chunk_refs(order=order, max_chunks_per_frame=max_chunks_per_frame):
            with open(ref.path, "rb") as f:
                data = f.read()
            yield HoloTVChunk(
                ref.frame_idx,
                ref.content_id,
                ref.chunk_id,
                data,
                ref.path,
                is_recovery=ref.is_recovery,
            )

    def iter_datagrams(
        self,
        *,
        order: str = "round_robin",
        max_chunks_per_frame: Optional[int] = None,
        repeats: int = 1,
        max_payload: int = 1200,
        auth_key: Optional[bytes] = None,
        enc_key: Optional[bytes] = None,
        key_id: int = 0,
    ) -> Iterator[bytes]:
        repeats = max(1, int(repeats))
        for chunk in self.iter_chunks(order=order, max_chunks_per_frame=max_chunks_per_frame):
            for _ in range(repeats):
                for datagram in iter_chunk_datagrams(
                    chunk.content_id,
                    chunk.chunk_id,
                    chunk.data,
                    max_payload=max_payload,
                    auth_key=auth_key,
                    enc_key=enc_key,
                    key_id=key_id,
                ):
                    yield datagram


class HoloTVReceiver:
    """
    Receive HoloTV datagrams and store chunks by frame content_id.
    """

    def __init__(
        self,
        stream_id: str,
        store: CortexStore,
        *,
        frame_indices: Optional[Iterable[int]] = None,
        auth_key: Optional[bytes] = None,
        enc_keys: Optional[dict[int, bytes]] = None,
        max_partial_age: float = 5.0,
    ) -> None:
        self.stream_id = str(stream_id)
        self.store = store
        self._assembler = ChunkAssembler(
            auth_key=auth_key,
            enc_keys=enc_keys,
            max_partial_age=max_partial_age,
        )
        self._content_to_frame: dict[bytes, int] = {}
        if frame_indices:
            for idx in frame_indices:
                self.register_frame(idx)

    def register_frame(self, frame_idx: int) -> bytes:
        cid = frame_content_id(self.stream_id, int(frame_idx))
        self._content_to_frame[cid] = int(frame_idx)
        return cid

    def content_id_for_frame(self, frame_idx: int) -> bytes:
        return frame_content_id(self.stream_id, int(frame_idx))

    def frame_idx_for_content_id(self, content_id: bytes) -> Optional[int]:
        return self._content_to_frame.get(content_id)

    def push_datagram(self, data: bytes) -> Optional[HoloTVReceipt]:
        assembled = self._assembler.push_datagram(data)
        if assembled is None:
            return None
        content_id, chunk_id, chunk_bytes = assembled
        stored = self.store.store_chunk_bytes(content_id, int(chunk_id), chunk_bytes)
        frame_idx = self._content_to_frame.get(content_id)
        return HoloTVReceipt(content_id, int(chunk_id), stored, frame_idx=frame_idx)

    def chunk_dir_for_frame(self, frame_idx: int) -> str:
        content_id = self.content_id_for_frame(frame_idx)
        return self.store.content_dir(content_id)

    def field_for_frame(self, frame_idx: int) -> "Field":
        if Field is None:
            raise NotImplementedError("Field helper not available in this build")
        content_id = f"{self.stream_id}::{int(frame_idx)}"
        return Field(content_id, self.chunk_dir_for_frame(frame_idx))
