# Transport Datagrams (UDP)

This document defines the UDP framing used by `holo.net.transport`.
It carries raw chunk bytes and optional control-plane messages.

## Chunk Datagram Format (HODT)
Header struct: `>4s16sIHHI` (big-endian)

```
0   4  magic      = "HODT"
4   16 content_id (16 bytes)
20  4  chunk_id   (u32)
24  2  frag_idx   (u16)
26  2  frag_total (u16)
28  4  chunk_len  (u32)
```

Payload: a fragment of the chunk bytes. `frag_total` fragments in total.
`chunk_len` is the full original chunk length.

`content_id` is fixed at 16 bytes. Use `digest_size=16` in
`content_id_bytes_from_uri`.

## Fragmentation
`iter_chunk_datagrams` splits `chunk_bytes` into fragments so that each
UDP datagram is at most `max_payload` bytes (including headers and optional
security overhead).

```
payload_size = max_payload - header_overhead
frag_total = ceil(len(chunk_bytes) / payload_size)
```

## Optional AES-GCM Encryption
If `enc_key` is provided:

- The datagram payload is replaced with:
  `key_id (1 byte) || nonce (12 bytes) || ciphertext+tag`.
- `nonce = blake2s(header || frag_idx, digest_size=12)`.
- AAD is `header || key_id`.

This is deterministic per fragment. AES-GCM requires `cryptography`.

## Optional HMAC Authentication
If `auth_key` is provided:

- Append `HMAC-SHA256` (32 bytes) over the full datagram bytes.
- If encryption is also enabled, HMAC is computed after encryption.

## Reassembly
`ChunkAssembler` groups datagrams by `(content_id, chunk_id)` and completes a
chunk when all fragments arrive. Partials expire after `max_partial_age`
(default 5.0s). If the assembled size does not match `chunk_len`, the chunk
is truncated to `chunk_len`.

## Control Datagrams
Control-plane INV/WANT messages use a separate format (`HOCT`). See:
`docs/spec/mesh-inv-want.md`.

## Related Code
- `src/holo/net/transport.py`
