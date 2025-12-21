# From Internet to HolographiX

## Thesis
The Internet assumes streams and sessions. HolographiX assumes a field: any
non-empty subset of chunks must decode into a coherent best-so-far estimate,
with quality improving as more chunks arrive.

Reliability is moved from the channel to the object. Transport is optional; the
representation carries the resilience.

## What This Enables
- Graceful loss: missing chunks reduce detail, not coordinates.
- Stateless decode: any subset works without coordination or sessions.
- Transport-agnostic delivery: UDP mesh, filesystems, object stores, or audio
  modems all carry the same chunk bytes.

## What This Repo Implements Today
- `holo.codec`: v2/v3 chunk formats for images and audio, plus scoring and
  optional recovery chunks.
- `holo.net.transport`: UDP datagram framing and reassembly.
- `holo.net.mesh`: minimal INV/WANT gossip and chunk exchange.
- `holo.tnc`: AFSK framing and WAV helpers for radio-style links.
- `holo.field`: local field decoding and healing.

## What This Repo Does Not Claim
- No global routing fabric, resolver chain, or content discovery network.
- No HTTP/WS kernel service; the interfaces are CLI and Python APIs.
- No identity, ACL, or privacy budget system beyond optional HMAC/AES-GCM.

If you need higher-level routing, policies, or governance, design them on top of
these primitives and keep them in the Wiki until they stabilize.
