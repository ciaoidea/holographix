<img width="36" height="36" alt="image" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX

| <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.17919892](https://doi.org/10.5281/zenodo.17919892) | <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source V1.4](https://github.com/ciaoidea/HolographiX) | <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/the-best-so-far-economy-why-i-m-betting-on-fields-not-streams-093b176be1e8) | <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast: [Dec 13th, 2025](https://github.com/user-attachments/assets/2fa1544d-888d-4774-993f-b3bf00da855e) |

<img width="1280" alt="image" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

HolographiX (Holographic Information MatriX) is an information layer for hostile, real-world environments—lossy radio links, mesh networks, mobile nodes, space links—and for adaptive systems that must act on incomplete evidence. It is meant as a shift in the information contract: from moving “streams that must arrive intact” to diffusing “fields of evidence that stay meaningful under damage”, across telecommunications, sensing/sensor-fusion, and semantic/agent networks.

The name is literal. “Holographic” means the object is represented so that each surviving fragment still carries globally useful information about the whole, the way a hologram remains recognizable even when you only have a piece. “Information MatriX” is not decoration: the X points to the deterministic index-space mapping that spreads evidence across a matrix of interchangeable contributions rather than keeping it trapped in a brittle stream—so resilience is engineered into representation, not outsourced to a perfect link.

Instead of treating data as a fragile stream that becomes meaningful only when complete and in-order, HolographiX turns an object into a population of interchangeable contributions (“holographic chunks”). Receive any non-empty subset and you can already form a coherent best-so-far estimate; receive more and it densifies smoothly. The target is not “deliver every bit reliably”, but maximum utility per surviving contribution under loss, burst loss, jitter, reordering, duplication, fading links, mobility, and intermittent connectivity.

This repository ships a concrete, measurable reference implementation for sensory media—RGB images and PCM WAV audio—because graceful degradation is visible, benchmarkable, and unforgiving. But the contract is general: any system where partial evidence should refine a state rather than stall can reuse the same idea, from telemetry that must remain actionable under dropouts to semantic/stateful pipelines where confidence should tighten as fragments arrive. Missing evidence should reduce fidelity or confidence, not force a stop.

The design is intentionally life-like in an engineering sense: information behaves like tissue. You can heal a damaged field by re-encoding the best current estimate into a fresh, well-distributed population, restoring robustness without pretending to resurrect lost truth. If you measure “intelligence” as the ability to anticipate and complete from incomplete cues, then healing is the infrastructure step toward that behavior: it keeps a coherent hypothesis alive as evidence arrives. The intelligence itself sits above the field—an agent or model that uses the field’s best-so-far state (and its uncertainty) to act and to predict what is missing.

---

## Try it in 60 seconds (from repo root)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src
python3 -m holo src/flower.jpg 8                   # encode → writes src/flower.jpg.holo
rm src/flower.jpg.holo/chunk_0000.holo            # simulate loss (optional)
python3 -m holo src/flower.jpg.holo --output src/flower_recon.png
````

Mental map: **MatriX codec** (coarse + residual, golden torsion interleave) → **transport** (chunk framing, UDP segmentation) → **field/mesh** (ingest, heal, gossip/INV/WANT, replication policy).

Quick shape (why it’s separable):

```text
[ MatriX codec ] --> holographic chunks --> [ transport ] --> datagrams --> [ field/mesh ] --> peers/gossip
        |                                     |                               |
 coarse + residual + index mixing             MTU-fit framing                 ingest + heal + INV/WANT
```

Packet-atomic note: keep `--packet-bytes` below path MTU (defaults to ~1168 bytes without HMAC). Small coarse thumbnails (`--coarse-side 16`) make each chunk fit one UDP datagram so links never fragment.

---

## The contract: from streams to fields (what HolographiX actually means)

Classic transport is endpoint-centric: a stream between two addresses that becomes meaningful only when completeness in order holds. That works for symbolic objects where “slightly wrong” often means “invalid”.

Perceptual (and adaptive) content lives in a different regime: structure is spread across space/time, and partial evidence can still be coherent. HolographiX makes that the specification: represent an object as a field of contributions such that almost any subset yields a globally consistent reconstruction whose quality grows smoothly with received evidence.

“Holographic Information MatriX” is meant literally: the object is not a fragile stream; it is a distributed population of evidence. The network is not asked to rescue a brittle representation. The representation is engineered so that survival of any subset remains meaningful.

---

## What is implemented right now

HolographiX is currently focused on sensory signals.

Images are handled as RGB arrays, with a model that generates a coarse approximation and an `int16` residual carrying fine detail.

Audio is handled as PCM WAV via the Python standard library `wave`, again with a coarse approximation plus an `int16` residual carrying fine detail.

Opaque arbitrary binaries are intentionally out of scope for now. Most binary formats are not meaningful under graceful blur and typically require strict erasure coding; HolographiX is about perceptual continuity (or confidence tightening) rather than bitwise equivalence under all failure modes.

---

## Layering model: MatriX, transport, field (and why it matters)

A useful way to read the repository is as three cleanly separated layers.

The `holo` codec produces holographic chunks: individually useful contributions that can be recombined from almost any subset. The MatriX part is the deterministic mixing rule that makes contributions interchangeable instead of local.

`holo.net` moves those chunks across harsh links. It frames, segments, reassembles, and keeps chunk identity separate from socket endpoints.

The HolographiX field layer (`holo.field`, with higher-level planning in `holo.net.mesh` and identity helpers in `holo.net.arch`) treats chunks as a shared substrate that many nodes can read and write, with local reconstruction and policy-driven healing.

A compact view of roles is captured by the mapping below; it is an analogy used as an engineering compass (not biology-as-physics):

| Component   | Engineering role                                                 | Informal mapping |
| ----------- | ---------------------------------------------------------------- | ---------------- |
| `codec`     | deterministic representation rules (formats, mixing, versioning) | genotype         |
| `field`     | best current reconstruction from surviving fragments             | phenotype        |
| `cortex`    | persistence, aging, deduplication, integrity checks              | tissue           |
| `mesh`      | circulation, gossip, opportunistic replication                   | ecology          |
| `arch`      | identity and compatibility (`holo://...` → content identifiers)  | receptors        |
| `transport` | UDP framing, segmentation, reassembly                            | impulses         |

<img width="800" alt="image" src="https://github.com/user-attachments/assets/ca097bb5-3aaa-4efa-ba5b-8e6495cbae44" />

The separation is deliberate. The codec does not depend on sockets. The transport does not depend on thumbnails or waveforms. The field logic does not depend on networking primitives. That boundary lets you change diffusion policy without touching MatriX mixing, and evolve models without rewriting packet transport.

---

## Beyond media: the same contract for adaptive intelligence (what “generalization” means here)

Even though the repository starts with images and audio, the core abstraction is not “a codec for JPEGs”. It is a field-centric representation-and-diffusion substrate.

Where a stream asks “did we receive the bytes yet?”, a field asks “did we receive enough evidence to act, and how much better did we get with the latest contribution?”. This maps onto adaptive systems: distributed robotics, multi-sensor fusion, edge inference, opportunistic agent networks, and model-centric pipelines where the “object” is a structured state (features, hypotheses, constraints, latent states) that should sharpen continuously as evidence arrives through unreliable channels.

This does not claim that today’s code “solves AI”. It claims something precise and testable: the information contract generalizes. Represent state as coarse + refinements; mix refinements so individual contributions are interchangeable; ingest contributions in any order while maintaining a coherent best-so-far estimate. If you want “intuitive completion”, you attach a predictor above the field that fills missing detail (with an explicit uncertainty model) and let healing regenerate a robust substrate for continued assimilation.

---

## The MatriX codec in one equation

Every encoded signal is split into a coarse component and a residual:

```text
residual = original - coarse_up
```

`coarse_up` is the coarse approximation upsampled back to the original resolution/length. The residual carries high-frequency detail. The codec stores the coarse representation plus a permuted, distributed residual across many chunks so that losing chunks reduces detail rather than invalidating the decode.

Decoding mirrors the same idea: reconstruct coarse; allocate a residual filled with zeros; write received residual samples into their positions; missing samples remain zero; add residual back to coarse with clipping. With all chunks present, reconstruction is exact or close (depending on model and settings). With a subset, reconstruction remains globally coherent and degrades smoothly.

---

## Golden-ratio interleaving as torsion/contorsion (interchangeability by construction)

If you cut the residual into contiguous blocks, you get brittle locality: lose one block and you lose one region or one time segment. HolographiX does the opposite. It treats the residual as a single line and “twists” it through index space so every chunk touches the whole signal.

The mixing primitive is a deterministic modular walk that behaves like a discrete torsion in index space:

```text
perm[i] = (i * step) mod N
```

The step is chosen near the golden step to avoid short periodic alignments, then minimally adjusted to guarantee a full cycle. That adjustment is the operational contorsion: the smallest correction that enforces complete coverage (a single orbit that visits every index exactly once).

The coprime constraint is:

```text
gcd(step, N) = 1
```

which guarantees the mapping is a full-cycle permutation over all indices.

If the residual is split into `B` chunks, chunk `b` takes a strided subsequence of the orbit:

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

Each chunk is therefore a phase slice of the same golden walk. Every chunk samples the whole signal in a quasi-uniform way instead of owning a local piece. When some chunks are lost, reconstruction degrades by losing global detail, not by punching holes in specific regions or time windows.

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
    codec.py              MatriX codec: chunk formats, headers, versioning,
                          compression and golden interleaving
    container.py          multi-object packing: one holographic store can
                          contain many objects
    field.py              local field for one content_id: ingest chunks,
                          track coverage, decode best view, perform healing
    cortex/               helpers for local storage; `store.py` is the backend
    models/               placeholder namespace for future signal models
    mind/                 mind-layer scaffold export; `dynamics.py` has z(t)
  net/                    networking namespace (`transport`, `arch`, `mesh`)

  codec_simulation/       React/Vite control deck that simulates codec
                          behavior, visualizes degradation, and generates
                          CLI commands
  docs/                   Guide to running a global holographic network
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

pip install -e ./src
```

Audio uses the standard library `wave`. Networking uses the standard library `socket` and `struct` plus the modules under `holo.net`.

---

## Quick start (CLI)

Encoding produces a `.holo` directory containing `chunk_XXXX.holo` files.

```bash
python3 -m holo --help
python3 -m holo src/flower.jpg 32
python3 -m holo src/flower.jpg.holo --output src/flower_recon.png

python3 -m holo /path/to/track.wav 32
python3 -m holo /path/to/track.wav.holo --output /path/to/track_recon.wav
```

To observe graded reconstruction, delete or move some `chunk_*.holo` files and decode again.

<img width="1280" alt="image" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" />

---

## Packet-native UDP: 1 datagram = 1 contribution

If you want the network to behave like a field of packets rather than a stream, keep every holographic chunk under your UDP payload budget so fragmentation never triggers. With default `max_payload=1200` and optional HMAC, the useful budget per datagram is approximately:

```text
payload_size = max_payload - 32 bytes of transport header - (32 bytes if HMAC)
```

Setting a small coarse thumbnail keeps each chunk comfortably inside that budget. Image encoding defaults to `--coarse-side 16`; the CLI also aims at packet-atomic chunks by default via `--packet-bytes 1168` (0 to disable).

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

Healing is policy, not magic. It does not recreate missing information. It takes the best currently reconstructable estimate and re-encodes it into a fresh holographic population, restoring a clean distribution of coarse and residual data so the field stays usable under long-lived impairment.

---

## Python API cheatsheet

```python
import holo
from holo.field import Field
from holo.net.arch import content_id_bytes_from_uri

holo.encode_image_holo_dir("frame.png", "frame.png.holo", target_chunk_kb=32)
holo.decode_image_holo_dir("frame.png.holo", "frame_recon.png")

holo.encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_recon.wav")

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

Above raw transport, `holo.net.mesh` adds gossip about which content IDs exist where and decides what to replicate and repeat. The intended style is that mesh policy remains small and explicit so different agents can adopt different replication strategies while reusing the MatriX codec and framing.

---

## Measuring resilience (turning intuition into curves)

A holographic layout is not a vibe; it is measurable. Fix an input signal, encode into `B` chunks, then for each `k` draw many random subsets of size `k`, decode, and measure quality against the original. For images, PSNR/MSE are a reasonable first pass. For audio, SNR is a baseline and perceptual measures can be added.

Two expected signatures indicate genuine interchangeability: mean quality improves smoothly with `k`, and variance across subsets at fixed `k` stays small. When those hold, quality depends mostly on how many fragments survived rather than on which specific identifiers survived.

---

## The `holo://` naming scheme

In HolographiX, `holo://...` is a content naming scheme used to derive a stable `content_id`. It is not a transport protocol. The scheme lets agents and tools refer to the same field identity independently of sockets, sessions, or endpoints.

```text
holo://object
```

From the library’s point of view, the entire string is opaque. The helper `content_id_bytes_from_uri` maps it deterministically to a fixed-length `content_id`. That `content_id`, plus a `chunk_id`, is what actually travels on the wire and what the mesh stores, gossips, and replicates.

---

## References

Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. [https://doi.org/10.1016/0001-6918(86)90062-4](https://doi.org/10.1016/0001-6918%2886%2990062-4)

Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4

Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge & Kegan Paul (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).

Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9

Rizzo, A. (2025). *HolographiX: Holographic Information Matrices for Robust Coding in Communication and Inference Networks* (v1.4.2). DOI: 10.5281/zenodo.17919892

Rizzo, A. (2025). *HolographiX: From Fragile Streams to Information Fields*. ISBN-13: 979-8278598534

<p align="center">
  © 2025 <a href="https://holographix.io">holographix.io</a>
</p>
