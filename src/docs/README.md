# HolographiX: Global Holographic Network Guide

This guide shows how to run HolographiX as a global “holographic network”: multiple nodes that encode content into holographic chunks, gossip those chunks over UDP (mesh/INV/WANT), and let any node reconstruct from whatever subset it receives. Everything below is self-contained and uses only what ships in this repository.

## Origin and radio lineage
- Conceived to evolve classic packet radio in the amateur world, building on earlier work by IK2TYW: same spirit of resilient, sessionless exchange over challenging links, but extended to chunk any payload and exploit modern mesh/DTN patterns.

## What you can build with HolographiX
- Resilient distribution of rich sensor payloads (any binary: waveforms, point clouds, maps, logs, parameters, media) that degrade gracefully under loss and improve as more chunks arrive.
- Global sensor fusion: scattered sites contribute to a shared field (astronomy synthetic aperture, seismic/geo awareness, weather/air-quality overlays, RF sensing, SAR/LiDAR mosaics).
- Live streams tolerant of delay/disconnect: cache-and-radiate DTN mesh over UDP, no central coordinator or sessions required.
- Map layers and overlays: render sensor data to tiles or vectors and spread them as chunked assets; partial tiles still show usable maps.
- Edge/IoT swarms: MTU-safe UDP chunks over constrained radios; opportunistic sync when peers meet; optional HMAC/AES-GCM for integrity/privacy.

## Example: global AI / neural network overlay
- **Roles**: sensors publish observations; edge/central workers subscribe, reconstruct, and run inference; outputs and updated models re-enter the mesh.
- **Content IDs** (suggested): `holo://obs/<domain>/<sensor>/<ts>` for observations, `holo://ai/model/<name>/<ver>` for weights/LoRA, `holo://ai/task/<id>` for task manifests, `holo://ai/out/<task-id>/<chunk>` for results.
- **Ship models over the mesh** (weights, shards, or LoRA deltas):
  ```bash
  python3 -m holo models/model_v2.bin 1 --packet-bytes 1000 --coarse-side 16
  CONTENT_URI="holo://ai/model/model_v2"
  ```
  Seed like any other content; nodes update once enough chunks arrive.
- **Inference flow**: (1) receive observation chunks and reconstruct locally, even if partial; (2) load the latest available model URI; (3) run inference; (4) chunk outputs (detections, embeddings, map overlays) back into the mesh under `holo://ai/out/...`.
- **Tasking/scheduling**: small JSON/CBOR manifests (`holo://ai/task/<id>`) describe what to run and with which model; multiple workers can respond, and consumers pick best/merge.
- **Training/federated updates**: share gradient/optimizer states or parameter deltas as chunked artifacts (`holo://ai/grad/<round>/<node>`); an aggregator node (or peer set) reconstructs and emits a new model URI for the next round.
- **Why HolographiX helps**: DTN-friendly cache-and-radiate means models/observations/results still propagate with loss/jitter/long RTT; any subset yields usable (if lower-fidelity) inputs and outputs instead of total failure.
- **When not to use**: tight RPC-style, low-latency control loops belong on a direct channel; use HolographiX for robust dissemination of observations, models, and batched outputs.

## Scope: why run it globally
- Build synthetic apertures from scattered observatories (ground + orbital): fuse high-res astronomy views as chunks arrive; tolerate long RTTs, loss, or intermittent contacts.
- Collect planet-scale seismic/geo telemetry: share seismic/infrasound/GNSS chunks across fault-line sites so partial coverage still yields usable situational awareness.
- Maintain multi-site perception for cameras/LiDAR/radar: a best-so-far field forms from whatever each site contributes; fidelity rises as more chunks arrive.
- Survive harsh links (loss, jitter, mobility) and opportunistic sync: no sessions needed; DTN-friendly gossip heals gaps when peers meet.
- Fit edge/IoT swarms: UDP + MTU-safe chunks over constrained radios; any subset reconstructs a usable view.
- Provide a substrate under apps/agents: not an AI network, but a robust information layer they can consume/produce.

## SNMP vs HolographiX (both ride UDP)
- **SNMP (manager/agent)**: best for control/monitoring (counters, state, traps). Assumes a reachable manager, no store-and-forward, payloads are small OIDs/values. Great to know if a station or sensor is alive and how it’s performing.
- **HolographiX (mesh/gossip)**: moves the actual observational data as chunks (not just status): seismic/acoustic waveforms, GNSS windows, radar/LiDAR cubes, point clouds, map tiles, log batches, parameter vectors, images/audio/video. Any subset is immediately useful; more chunks densify quality.
- **Resilience**: chunked gossip tolerates loss/jitter/RTT and DTN scenarios; nodes cache and re-radiate so late or intermittently connected peers still recover data. SNMP drops data if the manager can’t reach the agent in time.
- **Topology**: peer mesh (no single coordinator) vs hierarchical manager/agent.
- **Use together**: keep SNMP for health/config; run HolographiX for the rich sensor payloads when you need robustness and graceful degradation instead of binary “got it/lost it.”

## Example: seismic/geo sensing (maps and waveforms)
- **What to send**: seismic waveforms, infrasound, GNSS/IMU windows, accelerometer bursts. Treat each time window or tile as a content id.
- **Encode** each window into MTU-safe chunks (any binary works):
  ```bash
  python3 -m holo data/seismic_window.raw 1 --packet-bytes 1000 --coarse-side 16
  CONTENT_URI="holo://seismic/siteA/2024-04-01T12-00Z"
  ```
  This writes `data/seismic_window.raw.holo/` with chunks ready to seed.
- **Distribute** over the mesh (ground stations, ocean-bottom nodes, stratospheric or orbital relays):
  ```bash
  python3 src/examples/holo_mesh_node.py \
    --bind 0.0.0.0:5001 \
    --peer 198.51.100.2:5001 --peer 203.0.113.10:5001 \
    --store-root /var/hx_seismic \
    --max-payload 1000 \
    --seed-uri "$CONTENT_URI" \
    --seed-chunk-dir data/seismic_window.raw.holo \
    --frame-ttl 3600
  ```
  `--frame-ttl` keeps only the last hour of windows for live maps; set 0 to retain history.
- **Fuse into visual maps**: any node can decode partial windows and render heatmaps/contours; re-encode derived map tiles back into HolographiX for further sharing (`holo://seismic/map/<tile>`).
- **Why better than SNMP alone**: you move the waveform/tile data itself, not just device counters; late or lossy peers still get usable partial fields and progressively refine them.

## Example: geo maps / Google Earth overlays
- **What to send**: raster tiles (PNG/GeoTIFF), KMZ bundles, vector KML snippets, or pre-rendered overlays derived from sensors.
- **Encode a tile or KMZ** (any binary works):
  ```bash
  python3 -m holo data/tiles/z12_x2048_y1376.png 1 --packet-bytes 1000 --coarse-side 16
  CONTENT_URI="holo://maps/z12/2048/1376"
  # or bundle KML/KMZ: holo://maps/overlays/siteA.kmz
  ```
- **Distribute** the tile/overlay over the mesh (same mesh node command; change `--seed-uri`/`--seed-chunk-dir` accordingly). Nodes can cache and re-serve tiles even if intermittently connected.
- **Use in Google Earth**: decode the tile/KMZ to local disk and point Google Earth to it, or re-encode derived overlays (e.g., seismic heatmaps) back into HolographiX with URIs that match the tile grid.
- **Why it helps**: map layers remain shareable under lossy/DTN conditions; partial tile sets still render coarse views, and new tiles refine the map as they arrive.
- **Any sensor to map**: take any sensor feed (seismic, GNSS, weather, air quality, traffic, RF, astronomy, SAR/LiDAR), render it to raster tiles (heatmap, shaded relief) or vector layers (contours, points, polygons), then encode and distribute as above. Each tile/overlay is a chunkable asset; even partial coverage yields a usable map that sharpens as more tiles flow.

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
# writes src/flower.jpg.holo/ with chunk_XXXX.holo (+ .meta gain files); generated locally, not tracked
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
Only one node needs to seed; gossip will spread inventories and chunks. Ensure `src/flower.jpg.holo/` exists from the encode step above.
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
