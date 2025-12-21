# Test Vectors

All hex values are lowercase. Unless noted, `content_id` uses
`digest_size=16` and `str.strip()` normalization.

## Content IDs

Input:
- URI: `holo://demo/flower`
- URI (with spaces): `  holo://demo/flower  `
- stream_id: `stream-1`
- frame_idx: `7`

Expected:
- content_id(URI) = `c45da5fabae44317c8745e0879825bf9`
- content_id(URI with spaces) = `c45da5fabae44317c8745e0879825bf9`
- content_id(stream_id::frame_idx) = `6bb7b5784127a1326e5129ef8c658ce7`

## Control Datagrams (HOCT)

Parameters:
- content_id = `c45da5fabae44317c8745e0879825bf9`
- inventory chunk IDs = [0, 1, 2, 42]
- want chunk IDs = [1, 2]

Expected hex:
- encode_inventory_datagram =
  `484f435401c45da5fabae44317c8745e0879825bf90004000000010002002a`
- encode_want_datagram =
  `484f435402c45da5fabae44317c8745e0879825bf9000200010002`

## Chunk Datagrams (HODT)

Common parameters:
- content_id = `c45da5fabae44317c8745e0879825bf9`
- chunk_id = 7
- chunk_bytes = ASCII `hello` (hex `68656c6c6f`)

Plain (no auth, no encryption, max_payload=64):
- `484f4454c45da5fabae44317c8745e0879825bf900000007000000010000000568656c6c6f`

With HMAC (auth_key = `000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f`,
max_payload=128):
- `484f4454c45da5fabae44317c8745e0879825bf900000007000000010000000568656c6c6f2a6be14fb0380719756dfe673d6108b9d45110165813ac21e3ce2d0ea2fa481d`

With AES-GCM (enc_key = `000102030405060708090a0b0c0d0e0f`, key_id=3, max_payload=128):
- `484f4454c45da5fabae44317c8745e0879825bf900000007000000010000000503c30b9aca012eab82b37b929cd45bd36c7b3a9fb9338252e5515aa9a4f4d35a9d88`

## Recovery Chunk (HORC)

Parameters:
- slices = [01020304, 05060708]
- base_kind = 1 (image)
- base_codec_version = 3
- overhead = 0.5
- seed = 7

Expected hex:
- `484f52430101030000000002000000000000000400000002000000048b4af4aa6b7c`

Decoded header fields:
- block_count = 2
- coded_id = 0
- slice_len = 4
- coeff_len = 2
- payload_len = 4
