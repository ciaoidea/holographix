# HolographiX: Global Holographic Network Guide

This guide shows how to run HolographiX as a global “holographic network”: multiple nodes that encode content into holographic chunks, gossip those chunks over UDP (mesh/INV/WANT), and let any node reconstruct from whatever subset it receives. Everything below is self-contained and uses only what ships in this repository.

## Scope: why run it globally
- Build synthetic apertures from scattered observatories (ground + orbital): fuse high-res astronomy views as chunks arrive; tolerate long RTTs, loss, or intermittent contacts.
- Collect planet-scale seismic/geo telemetry: share seismic/infrasound/GNSS chunks across fault-line sites so partial coverage still yields usable situational awareness.
- Maintain multi-site perception for cameras/LiDAR/radar: a best-so-far field forms from whatever each site contributes; fidelity rises as more chunks arrive.
- Survive harsh links (loss, jitter, mobility) and opportunistic sync: no sessions needed; DTN-friendly gossip heals gaps when peers meet.
- Fit edge/IoT swarms: UDP + MTU-safe chunks over constrained radios; any subset reconstructs a usable view.
- Provide a substrate under apps/agents: not an AI network, but a robust information layer they can consume/produce.

## 1) Concepts (why it works)
- **Field, not stream**: content is a field of interchangeable contributions. Any subset of chunks yields a coherent reconstruction; more chunks densify quality.
- **Codec** (`holo.codec`): splits content into coarse + residual; distributes residual via golden-ratio interleave so every chunk touches the whole signal.
- **Transport** (`holo.net.transport`): frames chunks, segments to MTU-friendly datagrams, reassembles.
- **Mesh** (`holo.net.mesh.MeshNode` + `examples/holo_mesh_node.py`): listens on UDP, stores chunks, gossips inventories, forwards fresh chunks, re-radiates stored ones to grow coverage without sessions.
- **Cortex** (`holo.cortex.store.CortexStore`): persistent chunk store per content id; holds `.holo` files and optional per-chunk gain metadata.

## 2) Prerequisites
- Python 3.9+ recommended.
- From repo root:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e ./src
  ```
- Open UDP between peers (or tunnel via VPN/mesh overlay). Default commands use port 5001.
- Optional AES-GCM encryption requires `cryptography` (install via `pip install cryptography`).

## 3) Encode content for harsh links
Use packet-atomic settings so each chunk fits one UDP datagram (no fragmentation on bad links).
```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16
# writes src/flower.jpg.holo/ with chunk_XXXX.holo (+ .meta gain files)
```
Choose a content URI that identifies this object in the mesh:
```bash
CONTENT_URI="holo://demo/flower"
```

## 4) Bring up mesh nodes (local, LAN, or global)
Run this on every host, adjusting peers and paths. Works on loopback for testing or across the Internet if UDP is reachable.
```bash
python3 src/examples/holo_mesh_node.py \
  --bind 0.0.0.0:5001 \
  --peer 198.51.100.2:5001 --peer 203.0.113.10:5001 \
  --store-root /var/hx_cortex \
  --max-payload 1136 \
  --forward-prob 0.25 \
  --rate-hz 40 \
  --recency-bias 0.5 \
  --frame-ttl 0 \
  --auth-key deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef
```
Flags worth knowing:
- `--max-payload` <= path MTU; 1136 is safe on many links (IP+UDP overhead accounted for).
- `--forward-prob` probability to immediately forward a newly completed chunk.
- `--rate-hz` how often to re-radiate a random stored chunk.
- `--recency-bias` >0 favors newer content when picking a chunk to radiate.
- `--frame-ttl` purges old frames for time-coded content (seconds); use 0 to keep everything.
- `--auth-key` HMAC (hex, 32 bytes recommended). `--enc-key/--enc-key-id` adds AES-GCM.

### Seed the mesh with content
Only one node needs to seed; gossip will spread inventories and chunks.
```bash
python3 src/examples/holo_mesh_node.py \
  --bind 0.0.0.0:5001 \
  --peer 198.51.100.2:5001 \
  --store-root /var/hx_seed \
  --max-payload 1136 \
  --seed-uri "$CONTENT_URI" \
  --seed-chunk-dir src/flower.jpg.holo
```

### Decode anywhere
Once chunks flow, any node that has partial data can reconstruct:
```bash
python3 -m holo /var/hx_cortex/$CONTENT_URI --output /tmp/flower_recon.png
```
Missing chunks simply reduce detail; no brittle gaps.

## 5) Example topologies
- **Loopback demo** (single machine): run two terminals with `--bind 127.0.0.1:5001` / `5002` and cross-`--peer`; seed from one, decode on the other.
- **LAN mesh**: add `--mcast-group 239.1.1.1:5001` to exploit multicast; still list unicast peers for resilience.
- **Global over Internet**: open/forward UDP port on each site or use a VPN overlay (WireGuard/Tailscale/ZeroTier) so peers are routable; keep `--max-payload` conservative (e.g., 1024–1136).
- **Streaming frames**: derive content ids per frame using `holo.net.arch.content_id_bytes_from_stream_frame(stream_id, frame_idx)`; run `holo_mesh_node` with `--frame-ttl` to discard stale frames.

## 6) Operational tips
- **MTU**: stay below smallest hop; start with `--packet-bytes 1000` if unsure, then raise.
- **Health**: `MeshNode.counters` (exposed in code) track sent/stored/MAC failures; instrument if you wrap the example.
- **Storage hygiene**: use dedicated `--store-root` per node; TTL for live streams; `--recency-bias` to favor fresher content.
- **Security**: use `--auth-key` everywhere to reject forged chunks; add `--enc-key/--enc-key-id` for privacy; rotate keys by changing key id and re-encoding/re-seeding.
- **Reliability**: raise `--rate-hz` on high-loss paths; lower `--forward-prob` if you see congestion/duplicates; tune peers list so every site has at least two paths.

## 7) Systemd deployment (long-running nodes)
Templates live in `src/systemd/`:
- Edit `src/systemd/holo_mesh_node.env` (bind, store, peers, auth, extra flags).
- Place the env file at `/etc/default/holo_mesh_node` and the unit at `/etc/systemd/system/holo_mesh_node.service` (adjust `WorkingDirectory` to this repo or a pip-installed location).
- Reload and start:
  ```bash
  sudo systemctl daemon-reload
  sudo systemctl enable --now holo_mesh_node
  ```

## 8) Validation checklist
- Encode with `--packet-bytes` below MTU; verify directory `*.holo` exists.
- Start at least two nodes with reachable `--peer` links; confirm no socket bind errors.
- Seed from one node; watch other nodes populate `/var/hx_cortex/<content-id-hex>/chunk_*.holo`.
- Run decode; visual/audio output should improve as more chunks arrive (smooth degradation under loss).

With these steps, you have a reproducible pattern to stand up a holographic network on LANs, across the Internet, or atop any IP-reachable overlay. The mesh keeps content alive without sessions, and every surviving fragment remains globally useful.***
