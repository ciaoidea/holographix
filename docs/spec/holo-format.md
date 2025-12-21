# Holo Chunk and Container Formats

This document defines the on-disk chunk formats produced by `holo.codec` and
`holo.container`.

All multi-byte integers in headers are big-endian unless otherwise stated.
Residual slices are little-endian int16 arrays compressed with zlib.

## .holo Directory Layout
A chunk directory typically contains:

- `chunk_XXXX.holo`        Chunk payloads (the main format).
- `chunk_XXXX.holo.meta`   Optional ASCII float score (gain/energy).
- `recovery_XXXX.holo`     Optional RLNC recovery chunks.
- `manifest.json`          Optional manifest (see below).

`chunk_id` used in transport is derived from the filename. The `block_id` used
for reconstruction is stored inside the chunk header and MUST be trusted over
the filename. This matters for olonomic audio (v3), where chunks are named in
score order, not `block_id` order.

## Image Chunk Format (HOCH)
Header size: 30 bytes.

```
0   4  magic      = "HOCH"
4   1  version    = 1, 2, or 3
5   4  H          image height (u32)
9   4  W          image width  (u32)
13  1  C          channels (u8, currently 3)
14  4  block_count (u32)
18  4  block_id    (u32)
22  4  coarse_len  (u32)
26  4  resid_len   (u32)
```

Payload:
- `coarse_len` bytes of coarse payload.
- `resid_len` bytes of zlib-compressed residual slice.

### v1/v2 Image Payload
- Coarse payload is an encoded thumbnail (PNG/JPEG/WEBP).
- Residual slice is int16 (little-endian) of the pixel residuals.
- v1 uses strided residual placement. v2 uses golden permutation.

### v3 Image Payload (Olonomic)
Coarse payload begins with an `OLOI` metadata header:

```
OLOI_META_V2 (big-endian)
0   4  magic       = "OLOI"
4   1  meta_ver    = 2
5   1  block_size  DCT block size
6   1  quality     1..100
7   1  flags       bit0: coeffs clipped
8   2  pad_h       rows padded for block alignment
10  2  pad_w       cols padded for block alignment
12  4  payload_len length of coarse model payload
16  1  name_len    length of model name (ASCII)
17  N  name        coarse model name
17+N payload_len bytes of coarse model payload
```

- Residual slice contains quantized DCT coefficients (int16) in zigzag order.
- The full coefficient vector is permuted with the golden permutation, then
  sliced per block_id.
- Coarse model payload format is model-specific; see `holo.models.coarse`.

## Audio Chunk Format (HOAU)
Header size: 36 bytes.

```
0   4  magic       = "HOAU"
4   1  version     = 1, 2, or 3
5   1  channels    (u8)
6   1  sample_width (u8, always 2 for int16)
7   1  flags       bit0: residual clipped
8   4  sample_rate (u32)
12  4  n_frames    (u32)
16  4  block_count (u32)
20  4  block_id    (u32)
24  4  coarse_len  (u32)
28  4  coarse_size (u32)
32  4  resid_size  (u32)
```

Payload:
- `coarse_size` bytes of coarse payload.
- `resid_size` bytes of zlib-compressed residual slice.

### v1/v2 Audio Payload
- Coarse payload is zlib-compressed int16 samples of shape
  `(coarse_len, channels)`.
- Residual slice is int16 (little-endian) audio residuals.
- v1 uses strided residual placement. v2 uses golden permutation.

### v3 Audio Payload (Olonomic)
Coarse payload begins with an `OLOA` metadata header:

```
OLOA_META (big-endian)
0   4  magic     = "OLOA"
4   1  meta_ver  = 2
5   1  quality   1..100
6   1  flags     bit0: coeffs clipped
7   1  name_len  length of model name (ASCII)
8   4  n_fft
12  4  hop
16  N  name      coarse model name
16+N (rest)      coarse model payload
```

- Residual slice contains quantized STFT coefficients (int16), interleaved
  real/imag pairs per bin.
- The full coefficient vector is permuted with the golden permutation, then
  sliced per block_id.
- Coarse model payload format is model-specific; see `holo.models.coarse`.

## Chunk Manifests and Scores
If present, `manifest.json` for a simple chunk directory has this schema:

```
{
  "kind": "chunk_manifest",
  "manifest_version": 1,
  "base_kind": "image" | "audio",
  "codec_version": 2 | 3,
  "block_count": N,
  "ordered_chunks": ["chunk_0000.holo", ...],
  "chunks": [
    {"file": "chunk_0000.holo", "block_id": 0, "score": 123.0},
    ...
  ]
}
```

- Scores are stored in `chunk_*.holo.meta` as ASCII floats.
- `ordered_chunks` is used for gain-prioritized decoding or sending.

If `manifest.json` contains an `objects` array, it is a container manifest
(see below) and MUST NOT be parsed as a chunk manifest.

## Packed Container Format (HOPK)
`holo.container` packs multiple objects into one residual stream.

Container chunk header (`CONTAINER_HEADER`):

```
0   4  magic         = "HOPK"
4   1  version       = 1
5   4  block_count   (u32)
9   4  block_id      (u32)
13  8  total_residual (u64)
21  4  resid_size    (u32)
```

Payload:
- `resid_size` bytes of zlib-compressed int16 residual slice.

Container `manifest.json` schema (simplified):

```
{
  "version": 1,
  "block_count": N,
  "total_residual": M,
  "objects": [
    {
      "mode": "image" | "audio",
      "offset": ...,
      "residual_len": ...,
      "coarse_file": "obj_0000_coarse.bin",
      "version": 2,
      ...
    }
  ]
}
```

Each object references a `obj_XXXX_coarse.bin` file in the same directory.

## Related Code
- `src/holo/codec.py`
- `src/holo/container.py`
- `src/holo/models/coarse.py`
