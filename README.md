# <img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

HolographiX (HOLOGRAPHic Information matriX) is an information layer for hostile, real-world environments—lossy radio links, mesh networks, mobile nodes, space links—and for adaptive systems that must act on incomplete evidence. It is meant as a paradigm shift: from moving “streams that must arrive intact” to diffusing “fields of evidence that stay meaningful under damage”, across telecommunications, sensing/sensor-fusion, and semantic/agent networks.

The name is literal. “Holographic” means the object is represented so that each surviving fragment still carries globally useful information about the whole, the way a hologram remains recognizable even when you only have a piece. “Information matriX” points to the deterministic index-space mapping that spreads evidence across a matrix of interchangeable contributions rather than keeping it trapped in a brittle stream—so resilience is engineered into representation, not outsourced to a perfect link.

Instead of treating data as a fragile stream that only becomes meaningful when complete and in-order, HolographiX turns an object into a population of interchangeable contributions (“holographic chunks”). Receive any non‑empty subset and you can already form a coherent best‑so‑far estimate; receive more and it densifies smoothly. The target is not “deliver every bit reliably”, but maximum utility per surviving contribution under loss, burst loss, jitter, reordering, duplication, fading links, mobility, and intermittent connectivity.

The repository ships a concrete, measurable implementation for sensory media—RGB images and PCM WAV audio—because graceful degradation is visible, benchmarkable, and unforgiving. But the contract is general: any system where partial evidence should refine a state rather than stall can use the same idea, from telemetry that must remain actionable under dropouts to semantic/stateful pipelines where confidence should tighten as fragments arrive. Missing evidence should reduce fidelity or confidence, not force a stop.

The design is intentionally life-like in an engineering sense: information behaves like tissue. You can heal a damaged field by re‑encoding the best current estimate into a fresh, well‑distributed population, restoring robustness without pretending to resurrect lost truth. You can also stack multiple partial “exposures” to raise SNR and sharpen detail over time—extending naturally from images and audio to learned or conceptual states, where a vague hypothesis can condense into a stable attractor as evidence accumulates.



<video controls src="https://github.com/user-attachments/assets/....mp4"></video>



---

## Try it in 60 seconds (from repo root)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src
python3 -m holo src/flower.jpg 8                   # encode → writes src/flower.jpg.holo
rm src/flower.jpg.holo/chunk_0000.holo            # simulate loss (optional)
python3 -m holo src/flower.jpg.holo --output src/flower_recon.png
````

Mental map: **codec** (coarse + residual, golden torsion interleave) → **transport** (chunk framing, UDP segmentation) → **mesh** (gossip/INV/WANT, replication policy).

Quick shape (why it’s separable):

```text
[ codec ] --> holographic chunks --> [ transport ] --> datagrams --> [ mesh ] --> peers/gossip
   |                                     |                               |
 coarse + residual                       MTU-fit chunks                 INV/WANT + replication
```

Packet‑atomic note: keep `--packet-bytes` below path MTU (defaults to ~1168 bytes without HMAC). Small coarse thumbnails (`--coarse-side 16`) make each chunk fit one UDP datagram so links never fragment.

---

## The contract: from streams to fields (what HolographiX actually means)

Classic transport is endpoint‑centric: a stream between two addresses that becomes meaningful only when completeness in order holds. That works for symbolic objects where “slightly wrong” often means “invalid”.

Perceptual (and adaptive) content lives in a different regime: structure is spread across space/time, and partial evidence can still be coherent. HolographiX makes that the specification: represent an object as a **field of contributions** such that almost any subset yields a globally consistent reconstruction whose quality grows smoothly with received evidence.

“Holographic Information Matrix” is meant literally: the object is not a fragile stream; it is a distributed population of evidence. The network is not asked to rescue a brittle representation. The representation is engineered so that survival of *any subset* remains meaningful.

---

## What is implemented right now

HolographiX is currently focused on sensory signals:

Images are handled as RGB arrays, with a model that generates a coarse approximation and an `int16` residual carrying fine detail.

Audio is handled as PCM WAV via the Python standard library `wave`, again with a coarse approximation plus an `int16` residual carrying fine detail.

Opaque arbitrary binaries are intentionally out of scope for now. Most binary formats are not meaningful under graceful blur and typically require strict erasure coding; HolographiX is about perceptual continuity rather than bitwise equivalence under all failure modes.

---

## Layering model: codec, transport, field (and why it matters)

A useful way to read the repository is as three cleanly separated layers.

The `holo` codec produces *holographic chunks*: individually useful contributions that can be recombined from almost any subset.

`holo.net` moves those chunks across harsh links. It frames, segments, reassembles, and keeps chunk identity separate from socket endpoints.

The HolographiX “field” layer (`holo.field`, with higher‑level planning in `holo.net.mesh` and identity helpers in `holo.net.arch`) treats chunks as a shared substrate that many nodes can read and write, with local reconstruction and policy‑driven healing.

A compact view of roles is captured by the mapping below; it is an analogy used as an engineering compass, not as biology as physics:

| Component   | Engineering role                                                       | Informal mapping |
| ----------- | ---------------------------------------------------------------------- | ---------------- |
| `codec`     | deterministic representation rules (formats, interleaving, versioning) | genotype         |
| `field`     | best current reconstruction from surviving fragments                   | phenotype        |
| `cortex`    | persistence, aging, deduplication, integrity checks                    | tissue           |
| `mesh`      | circulation, gossip, opportunistic replication                         | ecology          |
| `arch`      | identity and compatibility (`holo://...` → content identifiers)        | receptors        |
| `transport` | UDP framing, segmentation, reassembly                                  | impulses         |

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/ca097bb5-3aaa-4efa-ba5b-8e6495cbae44" />

The separation is deliberate. The codec does not depend on sockets. The transport does not depend on thumbnails or waveforms. The field logic does not depend on networking primitives. That boundary lets you change diffusion policy without touching codec math, and evolve models without rewriting packet transport.

---

## Beyond media: the same contract for adaptive intelligence (what “generalization” means here)

Even though the repository starts with images and audio, the core abstraction is not “a codec for JPEGs”. It is a **field‑centric representation‑and‑diffusion substrate**.

Where a stream asks “did we receive the bytes yet?”, a field asks “did we receive enough evidence to act, and how much better did we get with the latest contribution?”. This maps onto adaptive systems: distributed robotics, multi‑sensor fusion, edge inference, opportunistic agent networks, and model‑centric pipelines where the “object” is a structured state (features, hypotheses, constraints, latent states) that should sharpen continuously as evidence arrives through unreliable channels.

This does not claim that today’s code “solves AI”. It claims something more precise and testable: the **information contract** generalizes. Represent state as coarse + refinements; mix refinements so individual contributions are interchangeable; ingest contributions in any order while maintaining a coherent best‑so‑far estimate.

---

## The holographic codec in one equation

Every encoded signal is split into a coarse component and a residual:

```text
residual = original - coarse_up
```

`coarse_up` is the coarse approximation upsampled back to the original resolution/length. The residual carries high‑frequency detail. The codec stores the coarse representation plus a permuted, distributed residual across many chunks so that losing chunks reduces detail rather than invalidating the decode.

Decoding mirrors the same idea: reconstruct coarse; allocate a residual filled with zeros; write received residual samples into their positions; missing samples remain zero; add residual back to coarse with clipping. With all chunks present, reconstruction is exact or close (depending on the selected model and compression settings). With a subset, reconstruction remains globally coherent and degrades smoothly.

---

## Golden‑ratio interleaving as torsion/contorsion (interchangeability by construction)

If you cut the residual into contiguous blocks, you get brittle locality: lose one block and you lose one region or one time segment. HolographiX does the opposite. It treats the residual as a single line and “twists” it through index space so every chunk touches the whole signal.

The mixing primitive is a deterministic modular walk that behaves like a discrete **torsion** (a spiral) in index space:

```text
perm[i] = (i * step) mod N
```

The step is chosen near the golden step to avoid short periodic alignments, then minimally adjusted to guarantee a full cycle. That adjustment is the operational **contorsion**: the smallest correction that enforces complete coverage (a single orbit that visits every index exactly once).

The golden ratio is introduced from the simplest geometric condition. Take a segment and split it into a larger and a smaller part. The golden condition is that the ratio of the whole to the larger part is the same as the ratio of the larger part to the smaller:

```text
whole : larger  =  larger : smaller
```

Set the whole to `1`, the larger part to `x`, and the smaller to `(1 − x)`. Then:

```text
1 / x = x / (1 − x)

x^2 + x − 1 = 0
x = (sqrt(5) − 1) / 2  ≈  0.618033...
```

The classical golden ratio `phi` is:

```text
phi     = (1 + sqrt(5)) / 2  ≈  1.618033...
phi - 1 = 1 / phi            ≈  0.618033...
```

Once the residual has been flattened into a 1‑D array of length `N`, the codec turns the golden fraction into a discrete rotation step:

```text
step ≈ (phi − 1) * N   (i.e. N / phi)
```

The contorsion constraint is the coprime condition:

```text
gcd(step, N) = 1
```

which guarantees the mapping is a single full‑cycle permutation over all indices.

If the residual is split into `B` chunks, chunk `b` takes a strided subsequence of the orbit:

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

Each chunk is therefore a phase slice of the same golden walk. Every chunk samples the whole signal in a quasi‑uniform way instead of owning a local piece. When some chunks are lost, reconstruction degrades by losing global detail, not by punching holes in specific regions or time windows.

The measurable claim is simple: quality should depend mainly on how many chunks arrived, not on which ids arrived. Good mixing yields low variance across random subsets of equal size and degrades without catastrophic discontinuities.

---

## Repository structure

```text
README.md                 this file
src/
  pyproject.toml          packaging entrypoint (editable install lives here)
  requirements.txt        optional pinned deps
  flower.jpg, galaxy.jpg,
  no-signal.jpg           sample media for quick tests

  holo/                   public API for image/audio encode-decode,
                          multi-object packing and Field
    __main__.py           CLI entry point: argument parsing and dispatch only
    codec.py              single-signal codec: chunk formats, headers,
                          versioning, compression and golden interleaving
    container.py          multi-object packing: one holographic store can
                          contain many objects
    field.py              local field for one content_id: ingest chunks,
                          track coverage, decode best view, perform healing
    cortex/               helpers for local storage; `store.py` is the backend
    models/               placeholder namespace for future signal models
    mind/                 mind-layer scaffold export; `dynamics.py` has z(t)
    net/                  networking namespace (`transport`, `arch`, `mesh`)

  codec_simulation/       React/Vite control deck that simulates codec
                          behavior, visualizes degradation, and generates
                          CLI commands
  examples/               ready-to-run demos
  infra/                  lab + containerlab material
  systemd/                sample systemd units
  tests/                  minimal test harness
```

---

## Installation

A recent Python 3 with NumPy and Pillow is sufficient for images. The packaging files live under `src/`.

```bash
git clone https://github.com/ciaoidea/holographix.git
cd holographix

python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -e ./src           # installs holo + dependencies from src/pyproject.toml
# optional dev tools: pip install -e ./src[dev]
```

Audio uses the standard library `wave`. Networking uses the standard library `socket` and `struct` plus the modules under `holo.net`. To run directly from the checkout without installing, add `src` to `PYTHONPATH` or `cd src` before invoking Python.

---

## Quick start (CLI)

Encoding produces a `.holo` directory containing `chunk_XXXX.holo` files. Sample media live in `src/`; either run the commands from inside `src` (add `PYTHONPATH=.` if you did not install) or reference the `src/...` paths from the repo root.

```bash
python3 -m holo --help
python3 -m holo src/flower.jpg
python3 -m holo src/flower.jpg 32
python3 -m holo src/flower.jpg.holo      # or: python3 -m holo src/flower.jpg.holo/
python3 -m holo /path/to/track.wav 32
python3 -m holo /path/to/track.wav.holo
```

To observe graded reconstruction, delete or move some `chunk_*.holo` files and decode again. The output should remain valid and globally coherent, with reduced detail.

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" />

### Quick profiles (encode/decode)

Pre‑tuned modes for mesh MTU and archive use.

| Profile             | When to use it                                                      | Encode command                                                                                                                                  | Decode command                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image-live-mesh`   | “immediately useful” image on bad/mesh links                        | `python3 -m holo INPUT.jpg 1 --packet-bytes 1136 --coarse-side 16`                                                                              | `python3 -m holo INPUT.jpg.holo --output OUT.png`                                                                                                                                                           |
| `image-archive`     | higher quality, fewer MTU limits (storage / non-real-time transfer) | `python3 -m holo INPUT.jpg 32 --packet-bytes 0 --coarse-side 64`                                                                                | `python3 -m holo INPUT.jpg.holo --output OUT.png`                                                                                                                                                           |
| `audio-live-mesh`   | robust audio (radio/mesh), low latency                              | `python3 -m holo INPUT.wav 2 --packet-bytes 1136`                                                                                               | `python3 -m holo INPUT.wav.holo --output OUT.wav`                                                                                                                                                           |
| `audio-archive`     | higher-fidelity audio, less overhead                                | `python3 -m holo INPUT.wav 32 --packet-bytes 0`                                                                                                 | `python3 -m holo INPUT.wav.holo --output OUT.wav`                                                                                                                                                           |
| `stack-images`      | “photon collector”: multiple exposures to raise SNR                 | `python3 -m holo FRAME1.jpg 8 && python3 -m holo FRAME2.jpg 8 && python3 -m holo FRAME3.jpg 8`                                                  | `python3 -m holo --stack FRAME1.jpg.holo FRAME2.jpg.holo FRAME3.jpg.holo --stack-max-chunks 16 --output STACK.png`                                                                                          |
| `video-frames-live` | video as independent frames (robust: no inter-frame dependencies)   | `ffmpeg -i input.mp4 -vf fps=10 frames/%06d.png && for f in frames/*.png; do python3 -m holo "$f" 1 --packet-bytes 1136 --coarse-side 16; done` | `mkdir -p recon && for d in frames/*.png.holo; do python3 -m holo "$d" --output "recon/$(basename "$d" .holo).png"; done && ffmpeg -framerate 10 -i recon/%06d.png -c:v libx264 -pix_fmt yuv420p recon.mp4` |

If you tell me the MTU/payload budget you need (e.g., 1200, 1000, 512 bytes) and target FPS for video, you can tune `--coarse-side` and `--chunk-kb` so you don’t have to guess.

---

## Packet‑native UDP: 1 datagram = 1 contribution

If you want the network to behave like a field of packets rather than a stream, keep every holographic chunk under your UDP payload budget so fragmentation never triggers (`frag_total = 1`). With the default `max_payload=1200` and optional HMAC, the useful budget per datagram is:

```text
payload_size = max_payload - 32 bytes of transport header - (32 bytes if HMAC)
```

Setting a small coarse thumbnail keeps each chunk comfortably inside that budget. Image encoding now defaults to `--coarse-side 16`; the CLI also aims at packet‑atomic chunks by default via `--packet-bytes 1168` (0 to disable). You can sanity check an encode with:

```python
import glob, os
mx = max(os.path.getsize(p) for p in glob.glob("frame.holo/chunk_*.holo"))
print(mx)
```

Stay below ~1168 bytes (no HMAC) or ~1136 bytes (with HMAC) to remain packet‑atomic. Drop `--coarse-side` further if needed. Expect a large number of tiny chunks for high‑res images; disable `--packet-bytes` if you prefer fewer, larger chunks.

`src/examples/holo_mesh_node.py` runs a minimal RX+TX+gossip loop: it stores completed chunks, forwards a fraction of fresh ones, and re‑sends random stored chunks at a fixed rate to keep coverage growing without sessions. It now speaks a light control plane (INV/WANT datagrams) to reduce duplicates and ask for missing chunks, weighs what to radiate based on per‑chunk gain (saved in `chunk_XXXX.holo.meta` during encode) and peer rarity, supports recency bias and TTL for time‑ordered IDs (`--recency-bias`, `--frame-ttl`), can use multicast in LAN (`mcast_groups`) alongside unicast peers, and can optionally encrypt chunks with AES‑GCM (`--enc-key/--enc-key-id`, HMAC still supported). Run it from inside `src/` (or set `PYTHONPATH=src`) so imports resolve. For streaming, derive one content_id per frame via `content_id_bytes_from_stream_frame(stream_id, frame_idx)`.

Coarse is pluggable: choose the coarse thumbnail format at encode time (`--coarse-format png|jpeg|webp` and optional `--coarse-quality` for lossy formats). Decoder auto‑detects via Pillow.

Instrumentation: MeshNode tracks datagrams seen, chunks stored, duplicates, and per‑content chunk coverage (visible via its `stats` dict). You can hook this into your own logging to plot coverage/quality over time.

Crypto: HMAC remains supported; optional AES‑GCM encryption (`--enc-key/--enc-key-id` in the mesh node) needs `cryptography` installed if you enable it.

---

## Running as services (systemd)

Units live in `src/systemd/` and assume the repo is at `/opt/holographix/src` (Python at `/usr/bin/python3`). Copy the unit to `/etc/systemd/system/`, copy the matching `.env` to `/etc/default/...`, edit values, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now <unit name>
```

Mesh node (RX+TX+gossip): unit `src/systemd/holo_mesh_node.service`, env template `src/systemd/holo_mesh_node.env` → `/etc/default/holo_mesh_node`. Key vars: `HXN_BIND`, `HXN_STORE_ROOT`, `HXN_PEER_OPTS="--peer ip:port ..."`, optional `HXN_SEED_OPTS` (bootstrap content), `HXN_EXTRA_OPTS="--packet-bytes 1136 --coarse-side 16"` for packet‑atomic behavior, `HXN_AUTH_OPTS` for HMAC. The node listens, stores chunks, forwards a slice of fresh ones, and periodically re‑radiates random stored chunks to peers.

Sender / receiver (one‑shot scripts): sender `src/systemd/holo_mesh_sender.service` + `src/systemd/holo_mesh_sender.env` (set `HXS_URI`, `HXS_CHUNK_DIR`, `HXS_PEER`, `HXS_MAX_PAYLOAD`, optional auth/extra). Receiver `src/systemd/holo_mesh_receiver.service` + `src/systemd/holo_mesh_receiver.env` (set `HXR_LISTEN`, `HXR_OUT_DIR`, `HXR_DECODE`, `HXR_MAX_PAYLOAD`, optional auth/extra).

Adjust paths inside the units if your install lives elsewhere; with the current layout, set `PYTHONPATH` and `WorkingDirectory` to your `/opt/holographix/src` checkout or install the package in the system Python.

---

## Codec simulation UI (control deck)

`src/codec_simulation/` contains an interactive React/Vite deck to poke the codec, visualize degradation, and emit equivalent CLI commands. It runs entirely in Node.js locally via the Vite dev server; an Electron shell is optional and not required.

```bash
cd src/codec_simulation
npm install
npm run dev
npm run build
```

---

## Perceptual stacking (photon‑collector mode)

If you have several holographic exposures of the same scene (for example faint objects, low light, or noisy sensors), you can stack them to improve SNR over time, exactly as with exposure stacking in astrophotography.

At the Python level:

```python
from holo.codec import (
    encode_image_holo_dir,
    stack_image_holo_dirs,
)

encode_image_holo_dir("spacetime-frame-1.jpg", "frame 1", target_chunk_kb=32)
encode_image_holo_dir("spacetime-frame-2.jpg", "frame 2", target_chunk_kb=32)
encode_image_holo_dir("spacetime-frame-3.jpg", "frame 3", target_chunk_kb=32)

stack_image_holo_dirs(
    ["spacetime-frame-1.jpg.holo", "spacetime-frame-2.jpg.holo", "spacetime-frame-3.jpg.holo"],
    "stacked_recon.png",
    max_chunks=16,
)
```

`stack_image_holo_dirs` decodes each `.holo` directory, sums the images in float, and writes the pixel‑wise average as `stacked_recon.png`. Uncorrelated noise cancels out; persistent structure reinforces.

CLI stacking shortcut:

```bash
python3 -m holo --stack frame1.jpg.holo frame2.jpg.holo frame3.jpg.holo \
    --stack-max-chunks 16 \
    --output stacked_recon.png
```

<p align="center">
  <img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/c2b939d1-8911-4381-8bd7-a93e29f5401c" /><br/>
  <em>HolographiX photon-collector mode: building high-resolution reconstructions from multiple <code>holo://objectID</code> exposures.</em>
</p>

---

## Multi‑object holographic fields (tissue‑like layout)

When several images or audio tracks belong to the same conceptual object, you can store them in a single holographic field instead of separate directories. All objects share the same “tissue”: losing chunks reduces detail across the whole pack instead of killing one file while leaving another perfect.

Using the Python API (container layer):

```python
import holo

holo.pack_objects_holo_dir(
    ["flower.jpg", "galaxy.jpg", "track.wav"],
    "scene.holo",
    target_chunk_kb=32,
)

holo.unpack_object_from_holo_dir("scene.holo", 0, output_path="flower_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 1, output_path="galaxy_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 2, output_path="track_rec.wav")
```

Tip: choose an `output_path` extension that matches the object type. Saving an image to `.wav` (or vice versa) will raise an error.

In this layout the residuals of all objects live on a single golden‑ratio trajectory. Any surviving chunk contributes information about every object. If chunks are lost, all members of the pack become slightly blurrier or more lo‑fi, but all remain decodable.

---

## Fields and healing (local metabolism)

A `Field` instance tracks which chunks are present for a given `content_id`, reports coverage, and can decode the best current percept at any time.

```python
from holo.field import Field

f = Field(content_id="demo/image", chunk_dir="image.png.holo")

summary = f.coverage()
print("present blocks:", summary["present_blocks"], "out of", summary["total_blocks"])

img_path = f.best_decode_image()
print("best decode saved to", img_path)

f.heal_to("image_healed.holo", target_chunk_kb=32)
```

Healing is policy, not magic. It does not recreate missing information. It takes the best currently reconstructable estimate, re‑encodes it into a fresh holographic population, and restores a clean distribution of coarse and residual data so the field stays usable under long‑lived impairment.

---

## Python API cheatsheet

These mirror the CLI but expose finer control.

```python
import holo
from holo.field import Field
from holo.net.arch import content_id_bytes_from_uri

holo.encode_image_holo_dir("frame.png", "frame.png.holo", target_chunk_kb=32)
holo.decode_image_holo_dir("frame.png.holo", "frame_recon.png")

holo.encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_recon.wav")

from holo.codec import stack_image_holo_dirs
stack_image_holo_dirs(["t0.png.holo", "t1.png.holo"], "stacked.png", max_chunks=8)

holo.pack_objects_holo_dir(["image1.jpg", "image2.jpg", "track.wav"], "scene.holo", target_chunk_kb=32)
holo.unpack_object_from_holo_dir("scene.holo", 0, "image1_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 2, "track_rec.wav")

f = Field("demo/image", "frame.png.holo")
print(f.coverage())
f.best_decode_image()
f.heal_to("frame_healed.holo")

cid = content_id_bytes_from_uri("holo://demo/image")
```

---

## UDP transport and mesh

`holo.net.transport` frames each chunk with identifiers, then segments it into UDP datagrams and reassembles it on the receiver. It treats chunks as opaque bytes and does not depend on image/audio semantics.

A minimal sender:

```python
import socket
import glob
from holo.net.arch import content_id_bytes_from_uri
from holo.net.transport import send_chunk

content_uri = "holo://demo/example/image-0001"
content_id = content_id_bytes_from_uri(content_uri)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("192.168.1.50", 5000)

for chunk_id, path in enumerate(sorted(glob.glob("image.png.holo/chunk_*.holo"))):
    with open(path, "rb") as f:
        chunk_bytes = f.read()
    send_chunk(sock, addr, content_id, chunk_id, chunk_bytes, max_payload=1200, auth_key=b"optional-secret")
```

Sample loopback test (two terminals).

Terminal A (receiver):

```bash
python3 - <<'PY'
import socket, time
from holo.net.mesh import MeshNode
from holo.cortex.store import CortexStore

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5000))
sock.setblocking(False)
mesh = MeshNode(sock, CortexStore("cortex_rx"), auth_key=b"optional-secret")

print("listening on 127.0.0.1:5000")
t0 = time.time()
while time.time() - t0 < 5:
    res = mesh.recv_once()
    if res:
        cid, chunk_id, path = res
        print("stored", chunk_id, "->", path)
    time.sleep(0.05)
PY
```

Terminal B (sender):

```bash
python3 - <<'PY'
import socket
from holo.net.mesh import MeshNode
from holo.cortex.store import CortexStore

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mesh = MeshNode(sock, CortexStore("cortex_tx"), peers=[("127.0.0.1", 5000)], auth_key=b"optional-secret")
mesh.broadcast_chunk_dir("holo://demo/flower", "flower.jpg.holo", repeats=2)
print("sent")
PY
```

Above raw transport, `holo.net.mesh` adds gossip about which content IDs exist where and decides what to replicate and repeat. The intended style is that mesh policy remains small and explicit so different agents can adopt different replication strategies while reusing the same codec and framing. `MeshNode` accepts `auth_key` for HMAC verification, and `repeats` can be used to deliberately resend chunks on harsh links; counters for sent/stored chunks and MAC failures are exposed on `MeshNode.counters` and `ChunkAssembler.counters`.

A practical note for harsh links: UDP segmentation turns one logical chunk into many datagrams. On lossy links, “receive the entire chunk” can become significantly less likely than “receive most datagrams”. A field‑centric evolution path is therefore to make the smallest network contribution coincide with the smallest decodable contribution, so partial arrivals still improve the percept. The repository keeps codec and transport separate precisely to allow that evolution without entangling math and sockets.

---

## Examples (ready‑to‑run)

The `src/examples/` directory contains self‑contained scripts:

| Script                   | What it does                                                                                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `encode_and_corrupt.py`  | encodes `flower.jpg` (under `src/`), deletes random chunks, decodes to show graceful degradation (falls back to a generated gradient if missing)                                  |
| `pack_and_extract.py`    | packs `flower.jpg` and `galaxy.jpg` into one field, drops chunks, extracts one object from the damaged field (falls back to synthetic checker/stripes if missing)                 |
| `heal_demo.py`           | damages `no-signal.jpg`, then uses `Field.heal_to` to regenerate a clean holographic population (falls back to a synthetic target if missing)                                     |
| `mesh_loopback.py`       | simulates two UDP peers exchanging holographic chunks by `holo://` content ID using `galaxy.jpg` (falls back to a generated spiral if missing)                                    |
| `scene_stream_demo.py`   | emits periodic multi‑object frames (image + short audio) as containerised holographic chunks with deterministic `content_id` per frame, using MeshNode control‑plane for WANT/INV |
| `psnr_benchmark.py`      | encodes an image, randomly drops chunks, reports PSNR statistics as chunk count varies                                                                                            |
| `snr_benchmark_audio.py` | measures SNR vs received chunks for WAV (point it at your own `.wav` file)                                                                                                        |
| `timing_benchmark.py`    | measures encode/decode wall‑clock times                                                                                                                                           |
| `field_tools.py`         | lists, prunes, copies, merges `.holo` directories                                                                                                                                 |
| `infra/containerlab/`    | reproducible containerlab+FRR lab for HolographiX vs baseline comparisons with netem impairments                                                                                  |
| `holo/mind/`             | minimal `MindDynamics` stub evolving a latent `z(t)` with a Lyapunov‑like potential plus antisymmetric torsion `Ω`, ready to sit above Field observations                         |

Quick run (from repo root with editable install, or set `PYTHONPATH=src`):

```bash
python3 src/examples/encode_and_corrupt.py
python3 src/examples/pack_and_extract.py
python3 src/examples/heal_demo.py
python3 src/examples/mesh_loopback.py

python3 src/examples/scene_stream_demo.py send --peer 127.0.0.1:6000 --stream-id demo_scene --frames 4 --fps 2.0
python3 src/examples/scene_stream_demo.py recv --bind 0.0.0.0:6000 --stream-id demo_scene --frames 4 --duration 3 --decode-dir src/examples/out/scene_rx

python3 src/examples/psnr_benchmark.py --image src/flower.jpg --target-chunk-kb 32
python3 src/examples/snr_benchmark_audio.py --wav /path/to/track.wav --block-count 12
python3 src/examples/timing_benchmark.py --image src/flower.jpg --audio /path/to/track.wav
python3 src/examples/field_tools.py list src/flower.jpg.holo

containerlab deploy -t src/infra/containerlab/holo-lab.clab.yml
src/infra/containerlab/init_hosts.sh
```

Scripts look for sample images under `src/`; audio benchmarks need a WAV you provide. Outputs land in `src/examples/out/`. Each script prints the paths it writes so you can open them quickly.

---

## Measuring resilience (turning intuition into curves)

A holographic layout is not a vibe; it is measurable. Fix an input signal, encode into `B` chunks, then for each `k` in `[1..B]` draw many random subsets of size `k`, decode, and measure quality against the original. For images, PSNR/MSE are a reasonable first pass. For audio, SNR is a baseline and perceptual measures can be added if needed.

Two expected signatures indicate genuine interchangeability: mean quality improves smoothly with `k`, and variance across subsets at fixed `k` stays small. When those hold, quality depends mostly on how many fragments survived rather than on which specific identifiers survived—given the fixed coarse layout and coprime permutation, the residual variance is mainly driven by per‑chunk gain/importance (e.g., `chunk_XXXX.holo.meta`).

If you care about interaction realism (prosody, facial motion, affect), it is also worth measuring reconstruction stability as fragments arrive in time with burst loss and reordering. The goal is not only “good after enough data”, but “continuous without spurious discontinuities during acquisition”.

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/e8c700f2-e5b6-424b-a848-a230294e8269" />

---

## The `holo://` naming scheme in the HolographiX agent network

In HolographiX, `holo://...` is a **content naming scheme** used to derive a stable `content_id`. It is not a transport protocol. The scheme lets agents and tools refer to the same field identity independently of sockets, sessions, or endpoints.

```text
holo://object
```

From the library’s point of view, the entire string `holo://object` is opaque. The helper `content_id_bytes_from_uri` maps it deterministically to a fixed‑length `content_id`. That `content_id`, plus a `chunk_id`, is what actually travels on the wire and what the mesh stores, gossips, and replicates.

The “object” named here is whatever the system treats as one holographic content field: a single image, a stack of frames, an audio track, a packed group of media, or a derived product such as a global dust map or a learned state used by a model. All chunks that belong to that field share the same `content_id` and differ only by `chunk_id`.

A Mars rover might publish a navcam frame as:

```text
holo://mars/rover-7/navcam/frame/sol-1234
```

and a derived dust‑density field as:

```text
holo://mars/global/dust-field/daily-avg-2034-01-12
```

In both cases the pattern is the same: a name is turned into a `content_id`, and a population of holographic chunks for that `content_id` is diffused through the mesh. Any agent or model that knows the same `holo://object` name and receives some subset of its chunks can reconstruct a usable estimate because fine detail has already been spread holographically across the chunk population.

---

## Operational best practices

| Topic               | Practical guidance                                                                                                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunk sizing        | start around 16–32 KB (`--target-chunk-kb`), adjust down on very lossy links to increase per‑datagram survival; adjust coarse size (`--coarse-side`, `--coarse-frames`) upward if coarse blur is too strong |
| Audio clipping      | residuals outside int16 are clipped and flagged; keep input WAV within int16 dynamic range to avoid lossy clipping                                                                                          |
| Healing / stacking  | periodically call `Field.heal_to` when fields age and chunks are lost; for low‑SNR capture, collect multiple exposures and run `stack_image_holo_dirs`                                                      |
| Transport integrity | if you need authenticated transport, pass `auth_key` (bytes) to `MeshNode` or directly to `iter_chunk_datagrams`/`send_chunk` to enable per‑datagram HMAC‑SHA256 checks                                     |
| Benchmarking        | use `src/examples/psnr_benchmark.py` and `src/examples/snr_benchmark_audio.py` to characterise graceful degradation; use `src/examples/timing_benchmark.py` to log latency on target hardware               |
| Field hygiene       | `src/examples/field_tools.py` can list, drop, copy, and merge chunks to keep stores clean or curate partial fields                                                                                          |

---

## Deployment notes

| Area             | Notes                                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Service wrappers | run sender/receiver under a supervisor (systemd/pm2) with restart‑on‑failure and log rotation                                                                                                                                  |
| systemd samples  | `src/systemd/holo_mesh_sender.service` and `src/systemd/holo_mesh_receiver.service` illustrate minimal units (adjust URIs, paths, peers; point `PYTHONPATH`/`WorkingDirectory` at your `src/` checkout or install the package) |
| Security         | avoid hardcoding `auth_key`; load from secrets manager or env var; add optional payload encryption if needed                                                                                                                   |
| Observability    | export counters from `MeshNode`/`ChunkAssembler` via a metrics endpoint (Prometheus/OpenTelemetry) and log structured events for chunk send/recv                                                                               |
| Packaging        | use `src/pyproject.toml` for pip install; build a minimal container image (python slim + repo) with entrypoints for `src/examples/holo_mesh_sender.py` / `src/examples/holo_mesh_receiver.py`                                  |
| Network tuning   | set `max_payload` below path MTU; adjust `repeats` and netem profiles per environment; consider light FEC/ARQ if links require it                                                                                              |
| Compose          | `src/docker-compose.yml` launches sender/receiver pairs (Holo + baseline) for quick local runs                                                                                                                                 |

Quick compose run:

```text
# Ensure inputs exist (or adjust paths in src/docker-compose.yml):
#   flower.jpg and flower.jpg.holo alongside src/docker-compose.yml
#   generate .holo if missing (from repo root): python3 -m holo src/flower.jpg 32
docker-compose -f src/docker-compose.yml up --build
```

---

## Conceptual lineage (kept explicit and testable)

HolographiX borrows language from biology—morphogenesis, fields, healing—because it describes a distributed pattern that remains recognisable under constant material loss. The implementation stays strictly within explicit data structures and deterministic reconstruction rules; no non‑material mechanism is assumed.

The use of golden‑ratio steps is an engineering technique for near‑uniform sampling under modular rotation, chosen to spread residual detail globally with low bookkeeping.

The project also adopts a methodological stance: the deepest design work happens at the level of chosen concepts and axioms (representation as fields, identity addressing, graded reconstruction) and is then tested by concrete experiments on impaired networks.

---

## References

Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. [https://doi.org/10.1016/0001-6918(86)90062-4](https://doi.org/10.1016/0001-6918%2886%2990062-4)
Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4
Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge & Kegan Paul (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).
Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9
Rizzo, A. *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. [DOI: 10.4236/apm.2023.139038](https://doi.org/10.4236/apm.2023.139038)
Rizzo, A. (2025). HolographiX: Holographic Information Matrices for Robust Coding in Communication and Inference Networks (v1.4.2). [DOI: 10.5281/zenodo.17919892](https://doi.org/10.5281/zenodo.17919892)

<p align="center">
  © 2025 <a href="https://holographix.io">HolographiX</a>
</p>
