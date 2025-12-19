"""
holo

Stable public API for the Holographix codec.

This package intentionally keeps a hard separation between:
- codec math + on-disk chunk format (holo.codec)
- transport framing (holo.net.*)
- field logic / healing policy (holo.field)
- storage backends (holo.cortex.*)

Only the image/audio single-object encode/decode API is considered stable right now.
Everything else is intentionally marked as work-in-progress and may change.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "encode_image_holo_dir",
    "encode_image_olonomic_holo_dir",
    "decode_image_holo_dir",
    "decode_image_olonomic_holo_dir",
    "encode_audio_holo_dir",
    "encode_audio_olonomic_holo_dir",
    "decode_audio_holo_dir",
    "decode_audio_olonomic_holo_dir",
    "detect_mode_from_extension",
    "detect_mode_from_chunk_dir",
    "pack_objects_holo_dir",
    "unpack_object_from_holo_dir",
    "Field",
    "CortexStore",
    "content_id_bytes_from_uri",
    "content_id_hex_from_uri",
    "MindDynamics",
]

__version__ = "0.2.0"


from .codec import (  # noqa: E402
    decode_audio_holo_dir,
    decode_audio_olonomic_holo_dir,
    decode_image_holo_dir,
    decode_image_olonomic_holo_dir,
    detect_mode_from_chunk_dir,
    detect_mode_from_extension,
    encode_audio_holo_dir,
    encode_audio_olonomic_holo_dir,
    encode_image_holo_dir,
    encode_image_olonomic_holo_dir,
)


try:
    from .net.arch import content_id_bytes_from_uri, content_id_hex_from_uri  # type: ignore
except Exception:  # pragma: no cover
    import hashlib

    def _norm_holo_uri(uri: str) -> str:
        return uri.strip()

    def content_id_bytes_from_uri(uri: str, *, digest_size: int = 16) -> bytes:
        u = _norm_holo_uri(uri).encode("utf-8", errors="strict")
        return hashlib.blake2s(u, digest_size=int(digest_size)).digest()

    def content_id_hex_from_uri(uri: str, *, digest_size: int = 16) -> str:
        return content_id_bytes_from_uri(uri, digest_size=digest_size).hex()


try:
    from .container import pack_objects_holo_dir, unpack_object_from_holo_dir  # type: ignore
except Exception:  # pragma: no cover
    def pack_objects_holo_dir(*args, **kwargs):
        raise NotImplementedError("container helpers are work in progress (container.py not present yet)")

    def unpack_object_from_holo_dir(*args, **kwargs):
        raise NotImplementedError("container helpers are work in progress (container.py not present yet)")


try:
    from .field import Field  # type: ignore
except Exception:  # pragma: no cover
    class Field:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Field is work in progress (field.py not present yet)")


try:
    from .cortex.store import CortexStore  # type: ignore
except Exception:  # pragma: no cover
    class CortexStore:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CortexStore is work in progress (cortex/store.py not present yet)")

try:
    from .mind import MindDynamics  # type: ignore
except Exception:  # pragma: no cover
    class MindDynamics:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MindDynamics is work in progress (mind/dynamics.py not present yet)")
