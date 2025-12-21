# Holo URI and Content IDs

This document defines how a `holo://` URI maps to a binary `content_id`.
The goal is a stable, deterministic mapping that is identical across
implementations.

## Normalization
Current normalization is intentionally minimal. The implementation is:

- Trim leading and trailing Unicode whitespace using `str.strip()`.
- Do not lowercase.
- Do not percent-decode.
- Do not collapse slashes or resolve `.`/`..` segments.

In other words, the normalized URI is the raw string after trimming.

## Content ID Derivation
`content_id` is the BLAKE2s digest of the normalized URI, encoded as UTF-8.

Pseudocode (default digest size):

```
uri_norm = uri.strip()
content_id = blake2s(utf8(uri_norm), digest_size=16)
```

- `digest_size` defaults to 16 bytes (128 bits).
- Transport framing in `holo.net.transport` assumes 16-byte content IDs.

Hex form is produced by `content_id.hex()`.

## Stream Frame Content IDs
For stream-like content, `content_id_bytes_from_stream_frame` builds a
synthetic URI string:

```
base = f"{stream_id}::{frame_idx}"
content_id = blake2s(utf8(base.strip()), digest_size=16)
```

`frame_idx` is an integer formatted in base-10.

## Related Code
- `src/holo/net/arch.py`
