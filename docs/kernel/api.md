# Kernel API (Current Surface)

There is no long-running HTTP/WS service in this repository. The current kernel
interfaces are:

## CLI
- `python -m holo`         Encode, decode, and heal .holo directories.
- `python -m holo net`     Transport and mesh helpers (datagrams, mesh loops).
- `python -m holo tnc-*`   AFSK WAV helpers (tnc-tx, tnc-rx, tnc-wav-fix).

## Python API
- `holo.codec`             Encode/decode image/audio chunks.
- `holo.field.Field`       Local field coverage, decoding, and healing.
- `holo.net.transport`     Datagram framing and reassembly.
- `holo.net.mesh.MeshNode` Minimal mesh exchange (INV/WANT + chunk send/receive).
- `holo.cortex.store.CortexStore`  Filesystem chunk store.
- `holo.tnc`               Modem and framing for audio links.

If you need a service boundary (HTTP/WS, gRPC, etc.), build it on top of these
APIs and keep it in the Wiki until it stabilizes.
