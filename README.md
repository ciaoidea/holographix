# <img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX


<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

HolographiX is a holographic, matrix-based media and networking substrate engineered for resilient, extreme connectivity.

HolographiX (“holographic information matrix”) is a field-centric codec and UDP substrate for sensory content—RGB images and PCM WAV audio—meant to keep *useful percepts* alive on networks that behave like reality: loss, jitter, reordering, duplication, fading links, mobility. The design target is not “reliable delivery of every bit”; it is “maximum perceptual utility per bit that survives”, under quickly changing conditions, with an anytime reconstruction path that improves as fragments arrive.

If you are building systems where perception must continue during impairment—robots, remote presence, ad‑hoc mesh links, disaster networks, radios—HolographiX treats the medium as damaged by default and makes degradation graceful instead of catastrophic. It is explicitly tuned for the continuous, redundant regime where Large Vision Models (LVMs) and Large Audio Models (LAMs) operate: missing evidence should reduce fidelity or confidence, not force a stall.

---

## Paradigm: from streams to fields

Classic transport abstractions are endpoint-centric: a stream between two addresses that aims for completeness in order and blocks when completeness cannot be guaranteed. That mental model is a good fit for programs and symbolic objects where “slightly wrong” often means “invalid”.

Perceptual content lives in a different regime. Images and audio have structure spread across space and time; you can lose samples and still have a coherent scene or phrase. HolographiX takes that as a specification: represent sensory content as a *field* such that almost any subset of contributions yields a globally consistent reconstruction whose quality grows smoothly with received information.

In HolographiX, the network does not “rescue” a brittle representation. The representation itself is built so that survival of *any subset* is meaningful.

---

## What is implemented (current scope)

HolographiX is currently focused on sensory signals:

Images are handled as RGB arrays, with a model that generates a coarse approximation and an `int16` residual that carries fine detail.

Audio is handled as PCM WAV via the Python standard library `wave`, again with a coarse approximation plus an `int16` residual carrying fine detail.

Opaque arbitrary binaries are intentionally out of scope for now. Most binary formats are not meaningful under graceful blur and typically require strict erasure coding; HolographiX is about perceptual continuity rather than bitwise equivalence under all failure modes.

---

## Layering model (codec, transport, field)

A useful way to read the repository is as three cleanly separated layers:

The `holo` codec produces *holographic chunks*: individually useful contributions that can be recombined from almost any subset.

`holo.net` moves those chunks across harsh UDP links. It frames, segments, reassembles, and keeps chunk identity separate from socket endpoints.

The HolographiX “field” layer (`holo.field`, with higher-level planning in `holo.net.mesh` and identity helpers in `holo.net.arch`) treats the set of chunks as a shared perceptual substrate that many nodes can read and write, with local reconstruction and policy-driven healing.

A compact view of roles is captured by the mapping below; it is an analogy used as an engineering compass, not as biology as physics:

| Component | Engineering role | Informal mapping |
|---|---|---|
| `codec` | deterministic representation rules (formats, interleaving, versioning) | genotype |
| `field` | best current reconstruction from surviving fragments | phenotype |
| `cortex` | persistence, aging, deduplication, integrity checks | tissue |
| `mesh` | circulation, gossip, opportunistic replication | ecology |
| `arch` | identity and compatibility (`holo://...` → content identifiers) | receptors |
| `transport` | UDP framing, segmentation, reassembly | impulses |

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/ca097bb5-3aaa-4efa-ba5b-8e6495cbae44" />


---

## The holographic codec in one equation

Every encoded signal is split into a coarse component and a residual:

```text
residual = original - coarse_up
```

`coarse_up` is the coarse approximation upsampled back to the original resolution/length. The residual carries high-frequency detail. The codec stores the coarse representation plus a permuted, distributed residual across many chunks so that losing chunks reduces detail rather than invalidating the decode.

Decoding is the same physical idea in reverse: reconstruct coarse; allocate residual filled with zeros; write received residual samples into their positions; missing samples remain zero; add residual back to coarse with clipping. With all chunks present, reconstruction is exact or close (depending on the selected model and compression settings). With a subset, reconstruction remains globally coherent and degrades smoothly.

---

## Golden-ratio interleaving (why it exists, what it guarantees)

If you cut the residual into contiguous blocks, you get brittle locality: lose one block and you lose one region or one time segment. HolographiX does the opposite. It treats the residual as a single line and walks it with a step that is “as incommensurate as possible” with the length, so that every chunk samples the entire signal. The way that step is chosen comes from the simplest geometric definition of the golden ratio.

Take a segment and split it into a larger and a smaller part. The golden condition is that the ratio of the whole to the larger part is the same as the ratio of the larger part to the smaller:

```text
whole : larger  =  larger : smaller
```

Set the whole to `1`, the larger part to `x`, and the smaller to `(1 − x)`. Then

```text
1 / x = x / (1 − x)

x^2 + x − 1 = 0
x = (sqrt(5) − 1) / 2  ≈  0.618033...
```

The classical golden ratio `phi` is

```text
phi     = (1 + sqrt(5)) / 2  ≈  1.618033...
phi - 1 = 1 / phi            ≈  0.618033...
```

This “most irrational” proportion is what HolographiX uses to spread fine detail as evenly as possible.

Once the residual has been flattened into a 1-D array of length `N`, the codec turns the golden fraction into a discrete rotation step:

```text
step ≈ (phi − 1) * N   (i.e. N / phi)
```

The step is then nudged until it is coprime with `N`:

```text
gcd(step, N) = 1
```

which guarantees that the mapping

```text
perm[i] = (i * step) mod N
```

is a single full-cycle permutation over all indices. The residual is not cut into blocks; it is threaded along this orbit.

If the residual is split into `B` chunks, chunk `b` takes a strided subsequence of the orbit:

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

Each chunk is therefore a phase slice of the same golden walk over the residual. Every chunk touches the whole signal in a quasi-uniform way instead of “owning” a local piece. When some chunks are lost, reconstruction degrades by losing global detail, not by punching holes in specific regions or time windows.

The practical claim you can measure is not mystical: quality should depend mainly on how many samples arrived, not on which specific chunk IDs arrived. A good golden interleaving yields low variance across random subsets of equal size and degrades without catastrophic discontinuities.

---

## Repository structure

```text
holo/
  __init__.py        public API for image/audio encode-decode,
                     multi-object packing and Field

  __main__.py        CLI entry point: argument parsing and dispatch only

  codec.py           single-signal codec:
                     chunk formats, headers, versioning,
                     compression and golden interleaving

  container.py       multi-object packing:
                     one holographic store can contain many objects

  field.py           local field for one content_id:
                     ingest chunks, track coverage, decode best view,
                     perform healing

  cortex/
    __init__.py      helpers for local storage
    store.py         persistent storage backend for chunk sets
    visual.py        convenience helpers for visual experiments

  models/
    __init__.py      registry that selects a signal model
    image.py         image model: coarse thumbnail + int16 residual
    audio.py         audio model: coarse subsampling + int16 residual

  net/
    __init__.py      networking namespace
    transport.py     UDP framing, segmentation, reassembly
    arch.py          helpers for holo:// URIs and content identifiers
    mesh.py          peer overlay, gossip and chunk replication policy

codec_simulation/    React/Vite control deck that simulates codec behavior,
                     visualizes degradation, and generates CLI commands
```

The separation is deliberate. The codec does not depend on sockets. The transport does not depend on thumbnails or waveforms. The field logic does not depend on networking primitives. That boundary is what lets you alter mesh policy without touching codec math, and evolve models without rewriting packet transport.

---

## Installation

A recent Python 3 with NumPy and Pillow is sufficient for images.

```bash
git clone https://github.com/ciaoidea/HolographiX.io.git
cd HolographiX.io

python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install numpy pillow
```

Audio uses the standard library `wave`. Networking uses the standard library `socket` and `struct` plus the modules under `holo.net`.

Packaging (editable install):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick start (CLI)

Encoding produces a `.holo` directory containing `chunk_XXXX.holo` files.

```bash
# show options
python3 -m holo --help

# encode an image with default chunk sizing (use one of the sample JPGs in repo root)
python3 -m holo flower.jpg

# encode with target chunk size around 32 KB
python3 -m holo flower.jpg 32

# decode from the holographic directory back to an image (trailing slash ok)
python3 -m holo flower.jpg.holo      # or: python3 -m holo flower.jpg.holo/

# audio (PCM WAV)
python3 -m holo track.wav 32
python3 -m holo track.wav.holo
```

To observe graded reconstruction, delete or move some `chunk_*.holo` files and decode again. The output should remain valid and globally coherent, with reduced detail.



<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" />


---

### Codec simulation UI (control deck)

`codec_simulation/` contains an interactive React/Vite deck to poke the codec, visualize degradation, and emit equivalent CLI commands. It runs entirely in Node.js locally via the Vite dev server; an Electron shell is optional and not required.

```bash
cd codec_simulation
npm install
npm run dev
# optional: build static bundle for local serving
npm run build
```

---


### Perceptual stacking (photon-collector mode)

If you have several holographic exposures of the same scene (for example faint objects, low light, or noisy sensors), you can stack them to improve SNR over time, exactly as with exposure stacking in astrophotography.

At the Python level:

```python
from holo.codec import (
    encode_image_holo_dir,
    stack_image_holo_dirs,
)

# encode several frames of the same scene into independent holographic fields
encode_image_holo_dir("spacetime-frame-1.jpg", "frame 1", target_chunk_kb=32)
encode_image_holo_dir("spacetime-frame-2.jpg", "frame 2", target_chunk_kb=32)
encode_image_holo_dir("spacetime-frame-3.jpg", "frame 3", target_chunk_kb=32)

# later, reconstruct and stack them as a "photon collector"
stack_image_holo_dirs(
    ["spacetime-frame-1.jpg.holo", "spacetime-frame-2.jpg.holo", "spacetime-frame-3.jpg.holo"],
    "stacked_recon.png",
    max_chunks=16,   # optional: limit chunks per exposure
)
```

`stack_image_holo_dirs` decodes each `.holo` directory, sums the images in float, and writes the pixel-wise average as `stacked_recon.png`. Uncorrelated noise cancels out; persistent structure reinforces.

<p align="center">
  <img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/c2b939d1-8911-4381-8bd7-a93e29f5401c" /><br/>
  <em>HolographiX photon-collector mode: building high-resolution reconstructions from multiple <code>holo://objectID</code> exposures.</em>
</p>


---

### Multi-object holographic fields (tissue-like layout)

When several images or audio tracks belong to the same conceptual object, you can store them in a single holographic field instead of separate directories. All objects then share the same “tissue”: losing chunks reduces detail across the whole pack instead of killing one file while leaving another perfect.

Using the Python API (container layer):

```python
import holo

# pack several objects into one holographic field
holo.pack_objects_holo_dir(
    ["flower.jpg", "galaxy.jpg", "track.wav"],
    "scene.holo",
    target_chunk_kb=32,
)

# later, reconstruct individual objects by index
holo.unpack_object_from_holo_dir("scene.holo", 0,
                                 output_path="flower_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 1,
                                 output_path="galaxy_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 2,
                                 output_path="track_rec.wav")
```
Tip: choose an `output_path` extension that matches the object type. Saving an image to `.wav` (or vice versa) will raise an error.

In this layout the residuals of all objects live on a single golden-ratio trajectory. Any surviving chunk contributes information about every object. If chunks are lost, all members of the pack become slightly blurrier or more lo-fi, but all remain decodable. The field behaves like a shared perceptual tissue rather than a bag of independent files.

---

## Fields and healing (local metabolism)

A `Field` instance tracks which chunks are present for a given `content_id`, reports coverage, and can decode the best current percept at any time.

```python
from holo.field import Field

f = Field(content_id="demo/image", chunk_dir="image.png.holo")

summary = f.coverage()
print("present blocks:", summary["present_blocks"],
      "out of", summary["total_blocks"])

img_path = f.best_decode_image()  # writes image.png_recon.png
print("best decode saved to", img_path)

f.heal_to("image_healed.holo", target_chunk_kb=32)
```

Healing is policy, not magic. It does not recreate missing information. It takes the best currently reconstructable percept, re-encodes it into a fresh holographic population, and restores a clean distribution of coarse and residual data. The purpose is to prevent slow entropic decay when fragments are lost over time and to keep the field usable under long-lived impairment.

The intuitive picture is “grow new cells from the surviving tissue”: decoding with partial chunks gives you a coherent but lower-detail percept; healing then *increases* the chunk population again by re-encoding that percept so the field stays dense. This is effectively a field-level interpolation: missing residual samples are zero-filled, but the coarse base and surviving residuals are redistributed into a full set of chunks. You can do this locally, or after pulling whatever fragments are available over UDP (via `holo://...` IDs) to rebuild a higher-resolution view from a sparse network harvest.

---

## Python API cheatsheet

Use these as starting points; they mirror the CLI but expose finer control.

```python
import holo
from holo.field import Field
from holo.net.arch import content_id_bytes_from_uri

# Single-object encode / decode
holo.encode_image_holo_dir("frame.png", "frame.png.holo", target_chunk_kb=32)
holo.decode_image_holo_dir("frame.png.holo", "frame_recon.png")
holo.encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_recon.wav")

# Photon-collector stacking
from holo.codec import stack_image_holo_dirs
stack_image_holo_dirs(["t0.png.holo", "t1.png.holo"], "stacked.png", max_chunks=8)

# Multi-object packing in one field
holo.pack_objects_holo_dir(["image1.jpg", "image2.jpg", "track.wav"], "scene.holo", target_chunk_kb=32)
holo.unpack_object_from_holo_dir("scene.holo", 0, "image1_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 2, "track_rec.wav")

# Field coverage + healing
f = Field("demo/image", "frame.png.holo")
print(f.coverage())
f.best_decode_image()                 # writes frame_recon.png
f.heal_to("frame_healed.holo")        # re-encodes current best view

# Build a content identifier (for transport/mesh)
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

Sample loopback test for the mesh helper (two terminals):

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

Above raw transport, `holo.net.mesh` adds gossip about which content IDs exist where and decides what to replicate and repeat. The intended style is that mesh policy remains small and explicit so different agents can adopt different replication strategies while reusing the same codec and framing.
`MeshNode` accepts `auth_key` for HMAC verification, and `repeats` can be used to deliberately resend chunks on harsh links; basic counters for sent/stored chunks and MAC failures are exposed on `MeshNode.counters` and `ChunkAssembler.counters`.

A practical note for harsh links: UDP segmentation turns one logical chunk into many datagrams. On lossy links, “receive the entire chunk” can become significantly less likely than “receive most datagrams”. A field-centric evolution path is therefore to make the smallest network contribution coincide with the smallest decodable contribution, so partial arrivals still improve the percept. The repository keeps codec and transport separate precisely to allow that evolution without entangling math and sockets.

---

## Examples (ready-to-run)

The `examples/` directory contains self-contained scripts:

- `encode_and_corrupt.py` encodes `flower.jpg` (repo root), deletes random chunks, and decodes to show graceful degradation (falls back to a generated gradient if missing).
- `pack_and_extract.py` packs `flower.jpg` and `galaxy.jpg` into one field, drops chunks, and extracts one object from the damaged field (falls back to synthetic checker/stripes if missing).
- `heal_demo.py` damages `no-signal.jpg`, then uses `Field.heal_to` to regenerate a clean holographic population (falls back to a synthetic target if missing).
- `mesh_loopback.py` simulates two UDP peers exchanging holographic chunks by `holo://` content ID using `galaxy.jpg` (falls back to a generated spiral if missing).
- `psnr_benchmark.py` encodes an image, randomly drops chunks, and reports PSNR statistics as chunk count varies.
- `snr_benchmark_audio.py` measures SNR vs received chunks for WAV.
- `timing_benchmark.py` measures encode/decode wall-clock times.
- `field_tools.py` lists, prunes, copies, and merges `.holo` directories.
- `infra/containerlab/` hosts a reproducible containerlab+FRR lab for HolographiX vs baseline comparisons with netem impairments.

Quick run (from repo root):

```bash
# graceful degradation on a synthetic gradient
python3 examples/encode_and_corrupt.py

# shared tissue packing + damaged extraction
python3 examples/pack_and_extract.py

# healing after chunk loss
python3 examples/heal_demo.py

# local UDP loopback using content IDs (holo://demo/galaxy)
python3 examples/mesh_loopback.py

# PSNR vs chunk count (writes results to stdout)
python3 examples/psnr_benchmark.py --image flower.jpg --target-chunk-kb 32

# Audio SNR vs chunk count
python3 examples/snr_benchmark_audio.py --wav examples/data/track.wav --block-count 12

# Encode/decode timing (image and optional audio)
python3 examples/timing_benchmark.py --image flower.jpg --audio examples/data/track.wav

# Field inspection / curation
python3 examples/field_tools.py list flower.jpg.holo

# containerlab lab (see infra/containerlab/README.md)
containerlab deploy -t infra/containerlab/holo-lab.clab.yml
infra/containerlab/init_hosts.sh
```

What you get:
- `encode_and_corrupt.py` → `examples/out/flower.holo` and `flower_recon.png` (or `gradient.*` if the sample is missing).
- `pack_and_extract.py` → `examples/out/scene.holo` (packed), plus `examples/out/galaxy_recon.png` from the damaged field.
- `heal_demo.py` → `examples/out/no-signal_degraded.png`, `no-signal_healed.holo`, `no-signal_healed.png`.
- `mesh_loopback.py` → sender writes `examples/out/galaxy.holo`, receiver reconstructs `examples/out/galaxy_mesh_recon.png` addressed by `holo://demo/galaxy`.

All inputs the scripts need are under `examples/data/`; outputs land in `examples/out/`. Each script prints the paths it writes so you can open them quickly.

---

## Measuring resilience (turning intuition into curves)

A holographic layout is not a vibe; it is measurable. Fix an input signal, encode into `B` chunks, then for each `k` in `[1..B]` draw many random subsets of size `k`, decode, and measure quality against the original. For images, PSNR/MSE are a reasonable first pass. For audio, SNR is a baseline and perceptual measures can be added if needed.

Two expected signatures indicate genuine interchangeability: mean quality improves smoothly with `k`, and variance across subsets at fixed `k` stays small. When those hold, quality depends mostly on how many fragments survived rather than on which specific identifiers survived.

If you care about interaction realism (prosody, facial motion, affect), it is also worth measuring reconstruction stability as fragments arrive in time with burst loss and reordering. The goal is not only “good after enough data”, but “continuous without spurious discontinuities during acquisition”.

<img width="1280" height="800" alt="image" src="https://github.com/user-attachments/assets/e8c700f2-e5b6-424b-a848-a230294e8269" />

---


## `The holo:// protocol in the HolographiX agent network

In HolographiX, a network of agents and Large Media Models talks by reading and writing shared holographic fields. `holo://` is the scheme; the part that follows is the application-level name of one such field:

```text
holo://object
````

From the library’s point of view, the entire string `holo://object` is opaque. The helper `content_id_bytes_from_uri` maps it deterministically to a fixed-length `content_id`. That `content_id`, plus a `chunk_id`, is what actually travels on the wire and what the mesh stores, gossips, and replicates.

The “object” named here is whatever the system treats as one holographic content field. It can be a single image, a stack of frames, an audio track, a packed group of media, or a derived product such as a global dust map or a learned state that a Large Media Model uses internally. All chunks that belong to that field share the same `content_id` and differ only by `chunk_id`.

For example, a Mars rover might publish a navcam frame as

```text
holo://mars/rover-7/navcam/frame/sol-1234
```

and a derived dust-density field as

```text
holo://mars/global/dust-field/daily-avg-2034-01-12
```

In both cases the agent network sees the same pattern: a name `holo://object` is turned into a `content_id`, and a population of holographic chunks for that `content_id` is diffused through the mesh. Any agent or Large Media Model that knows the same `holo://object` name and receives some subset of its chunks can reconstruct a usable percept or state for that object, because fine detail has already been spread holographically across the chunk population.


---

## Operational best practices

- Chunk sizing: start around 16–32 KB (`--target-chunk-kb`), adjust down on very lossy links to increase per-datagram survival; adjust coarse size (`--coarse-side`, `--coarse-frames`) upward if coarse blur is too strong.
- Audio clipping: residuals outside int16 are clipped and flagged; keep input WAV within int16 dynamic range to avoid lossy clipping.
- Healing/stacking: periodically call `Field.heal_to` when fields age and chunks are lost; for low-SNR capture, collect multiple exposures and run `stack_image_holo_dirs`.
- Transport integrity: if you need authenticated transport, pass `auth_key` (bytes) to `MeshNode` or directly to `iter_chunk_datagrams`/`send_chunk` to enable per-datagram HMAC-SHA256 checks.
- Benchmarking: use `examples/psnr_benchmark.py` and `examples/snr_benchmark_audio.py` to characterise graceful degradation; use `examples/timing_benchmark.py` to log latency on target hardware.
- Field hygiene: `examples/field_tools.py` can list, drop, copy, and merge chunks to keep stores clean or to curate partial fields.

## Deployment notes

- Service wrappers: run sender/receiver under a supervisor (systemd/pm2) with restart-on-failure and log rotation.
- systemd samples: `systemd/holo_mesh_sender.service` and `systemd/holo_mesh_receiver.service` illustrate minimal units (adjust URIs, paths, peers).
- Security: avoid hardcoding `auth_key`; load from secrets manager or env var; add optional payload encryption if needed.
- Observability: export counters from `MeshNode`/`ChunkAssembler` via metrics endpoint (Prometheus/OpenTelemetry) and log structured events for chunk send/recv.
- Packaging: use `pyproject.toml` for pip install; build a minimal container image (python slim + repo) with entrypoints for `examples/holo_mesh_sender.py` / `holo_mesh_receiver.py`.
- Network tuning: set `max_payload` below path MTU; adjust `repeats` and netem profiles per environment; consider light FEC/ARQ if links require it.
- Compose: `docker-compose.yml` launches sender/receiver pairs (Holo + baseline) for quick local runs.

Quick compose run:
```
# Ensure inputs exist (or adjust paths in docker-compose.yml):
#   flower.jpg and flower.jpg.holo alongside docker-compose.yml
#   generate .holo if missing: python3 -m holo flower.jpg 32
docker-compose up --build
```

---

## Conceptual lineage (kept explicit and testable)

HolographiX borrows language from biology—morphogenesis, fields, healing—because it describes a distributed pattern that remains recognisable under constant material loss. The implementation stays strictly within explicit data structures and deterministic reconstruction rules; no non-material mechanism is assumed.

The use of golden-ratio steps is an engineering technique for near-uniform sampling under modular rotation, chosen to spread residual detail globally with low bookkeeping.

The project also adopts a methodological stance: the deepest design work happens at the level of chosen concepts and axioms (representation as fields, identity addressing, graded reconstruction) and is then tested by concrete experiments on impaired networks.

---

## References

Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. https://doi.org/10.1016/0001-6918(86)90062-4  

Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4  

Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge & Kegan Paul (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).  

Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9  

Rizzo, A. *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. [DOI: 10.4236/apm.2023.139038](https://doi.org/10.4236/apm.2023.139038)

Rizzo, A. (2025). *HolographiX: a percept-first codec and network substrate for Large Vision Models and Large Audio Models over UDP (v1.0.1).* Zenodo. (v1.0.1). [10.5281/zenodo.17844824](10.5281/zenodo.17844824)

--

<p align="center">
  © 2025 <a href="https://holographix.io">HolographiX</a>
</p>
