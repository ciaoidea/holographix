# Mesh UDP Guide

This guide shows how to run the current UDP mesh implementation in this repository.
It uses the codec in `holo.codec`, the transport in `holo.net.transport`, and
`holo.net.mesh.MeshNode` or the example scripts in `src/examples/`.

## 1) Install
From repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src
```

(Alternatively: `PYTHONPATH=src python3 ...`.)

## 2) Encode Content
Encode an image into a `.holo` directory:

```bash
python3 -m holo src/flower.jpg --packet-bytes 1136 --coarse-side 16
CONTENT_URI="holo://demo/flower"
```

Notes:
- `--packet-bytes` is only applied to image encoding. For audio, use the
  positional `TARGET_KB` or `--blocks` to control chunk count.
- `--packet-bytes` is a per-chunk budget, not a UDP datagram size. If a chunk
  exceeds `--max-payload`, it will be fragmented into multiple datagrams.

## 3) Run a Mesh Node (RX+TX+gossip)
On each node:

```bash
python3 src/examples/holo_mesh_node.py \
  --bind 0.0.0.0:5001 \
  --peer 198.51.100.2:5001 --peer 203.0.113.10:5001 \
  --store-root /var/hx_cortex \
  --max-payload 1136 \
  --rate-hz 40 \
  --forward-prob 0.25
```

Optional integrity/encryption:

```bash
--auth-key <hex>        # HMAC-SHA256 key
--enc-key <hex>         # AES-GCM key (requires cryptography)
--enc-key-id <int>      # key-id byte (default 0)
```

## 4) Seed Content into the Store
Seed a node with a local chunk directory:

```bash
python3 src/examples/holo_mesh_node.py \
  --bind 0.0.0.0:5001 \
  --peer 198.51.100.2:5001 \
  --store-root /var/hx_seed \
  --seed-uri "$CONTENT_URI" \
  --seed-chunk-dir src/flower.jpg.holo
```

Seeding loads the chunks into the local store; the node will re-radiate them
over time based on `--rate-hz` and `--forward-prob`.

To push all chunks immediately to a peer, use the sender helper:

```bash
python3 src/examples/holo_mesh_sender.py \
  --uri "$CONTENT_URI" \
  --chunk-dir src/flower.jpg.holo \
  --peer 198.51.100.2:5001
```

## 5) Receive and Decode
Simple receiver (stores chunks and decodes on exit):

```bash
python3 src/examples/holo_mesh_receiver.py \
  --listen 0.0.0.0:5001 \
  --out-dir /tmp/hx_rx \
  --decode /tmp/flower_recon.png \
  --duration 10
```

Manual decode from a store directory:

```bash
python3 -m holo /tmp/hx_rx/<content-id-hex> --output /tmp/flower_recon.png
```

`CortexStore` uses `content_id.hex()` as the directory name.

## 6) Control Plane (INV/WANT)
`MeshNode` periodically exchanges inventory and requests missing chunks. This is
simple and limited to 16-bit chunk IDs. Recovery chunks (`recovery_*.holo`) are
not requested by INV/WANT and must be sent proactively.

## Related Code
- `src/examples/holo_mesh_node.py`
- `src/examples/holo_mesh_sender.py`
- `src/examples/holo_mesh_receiver.py`
- `src/holo/net/mesh.py`
