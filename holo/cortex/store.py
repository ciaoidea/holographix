"""
holo.cortex.store

Persistent storage backend for holographic chunks.
"""

from __future__ import annotations

import os
from typing import Iterable


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
