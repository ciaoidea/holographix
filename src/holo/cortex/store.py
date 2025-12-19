"""
holo.cortex.store

Persistent storage backend for holographic chunks.
"""

from __future__ import annotations

import os
import struct
import time
from typing import Iterable

from ..recovery import REC_HEADER, REC_MAGIC


class CortexStore:
    """
    File-system backed chunk store.

    Chunks are grouped under a directory named by the hex content_id.
    """

    def __init__(self, root: str = "cortex_store") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def content_dir(self, content_id: bytes) -> str:
        return os.path.join(self.root, content_id.hex())

    def store_chunk_bytes(self, content_id: bytes, chunk_id: int, chunk_bytes: bytes) -> str:
        """
        Persist one chunk and return the stored path.
        """
        cdir = self.content_dir(content_id)
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"chunk_{int(chunk_id):04d}.holo")
        if len(chunk_bytes) >= 4 and chunk_bytes[:4] == REC_MAGIC and len(chunk_bytes) >= REC_HEADER.size:
            try:
                (_magic, _rec_ver, _base_kind, _base_codec_ver, _flags,
                 _block_count, coded_id, _slice_len, _coeff_len, _payload_len) = REC_HEADER.unpack_from(chunk_bytes, 0)
                path = os.path.join(cdir, f"recovery_{int(coded_id):04d}.holo")
            except struct.error:
                pass
        with open(path, "wb") as f:
            f.write(chunk_bytes)
        return path

    def iter_chunks(self, content_id: bytes) -> Iterable[str]:
        """
        Yield chunk paths for a given content_id.
        """
        cdir = self.content_dir(content_id)
        if not os.path.isdir(cdir):
            return []
        return sorted(
            os.path.join(cdir, p)
            for p in os.listdir(cdir)
            if p.endswith(".holo")
        )

    def chunk_count(self, content_id: bytes) -> int:
        """
        Count stored chunks for content_id.
        """
        cdir = self.content_dir(content_id)
        if not os.path.isdir(cdir):
            return 0
        return sum(1 for name in os.listdir(cdir) if name.endswith(".holo"))

    def list_content_ids(self) -> Iterable[bytes]:
        """
        List content_ids present in the store.
        """
        if not os.path.isdir(self.root):
            return []
        out = []
        for name in os.listdir(self.root):
            cdir = os.path.join(self.root, name)
            if os.path.isdir(cdir):
                try:
                    out.append(bytes.fromhex(name))
                except ValueError:
                    continue
        return out

    def content_last_mtime(self, content_id: bytes) -> float:
        """
        Return the most recent mtime among chunks of a content_id.
        """
        cdir = self.content_dir(content_id)
        latest = 0.0
        if not os.path.isdir(cdir):
            return latest
        for name in os.listdir(cdir):
            if not name.endswith(".holo"):
                continue
            path = os.path.join(cdir, name)
            try:
                m = os.path.getmtime(path)
                latest = max(latest, m)
            except OSError:
                continue
        return latest

    def purge_older_than(self, ttl_sec: float) -> None:
        """
        Drop content whose latest chunk mtime is older than now - ttl_sec.
        """
        now = time.time()
        for cid in list(self.list_content_ids()):
            m = self.content_last_mtime(cid)
            if m and now - m > ttl_sec:
                self.drop_content(cid)

    def drop_content(self, content_id: bytes) -> None:
        """
        Remove all chunks associated with a content_id.
        """
        cdir = self.content_dir(content_id)
        if not os.path.isdir(cdir):
            return
        for name in os.listdir(cdir):
            try:
                os.remove(os.path.join(cdir, name))
            except OSError:
                pass
        try:
            os.rmdir(cdir)
        except OSError:
            pass
