# Holographix
Holographix is a holographic, matrix-based media and networking substrate engineered for resilient, extreme connectivity.

Holographix (“holographic information matrix”) is a field-centric codec and UDP substrate for sensory content—RGB images and PCM WAV audio—meant to keep *useful percepts* alive on networks that behave like reality: loss, jitter, reordering, duplication, fading links, mobility. The design target is not “reliable delivery of every bit”; it is “maximum perceptual utility per bit that survives”, under quickly changing conditions, with an anytime reconstruction path that improves as fragments arrive.

If you are building systems where perception must continue during impairment—robots, remote presence, ad‑hoc mesh links, disaster networks, radios—Holographix treats the medium as damaged by default and makes degradation graceful instead of catastrophic. It is explicitly tuned for the continuous, redundant regime where Large Vision Models (LVMs) and Large Audio Models (LVAs) operate: missing evidence should reduce fidelity or confidence, not force a stall.

---

## Paradigm: from streams to fields

Classic transport abstractions are endpoint-centric: a stream between two addresses that aims for completeness in order and blocks when completeness cannot be guaranteed. That mental model is a good fit for programs and symbolic objects where “slightly wrong” often means “invalid”.

Perceptual content lives in a different regime. Images and audio have structure spread across space and time; you can lose samples and still have a coherent scene or phrase. Holographix takes that as a specification: represent sensory content as a *field* such that almost any subset of contributions yields a globally consistent reconstruction whose quality grows smoothly with received information.

In Holographix, the network does not “rescue” a brittle representation. The representation itself is built so that survival of *any subset* is meaningful.

---

## What is implemented (current scope)

Holographix is currently focused on sensory signals:

Images are handled as RGB arrays, with a model that generates a coarse approximation and an `int16` residual that carries fine detail.

Audio is handled as PCM WAV via the Python standard library `wave`, again with a coarse approximation plus an `int16` residual carrying fine detail.

Opaque arbitrary binaries are intentionally out of scope for now. Most binary formats are not meaningful under graceful blur and typically require strict erasure coding; Holographix is about perceptual continuity rather than bitwise equivalence under all failure modes.

---

## Layering model (codec, transport, field)

A useful way to read the repository is as three cleanly separated layers:

The `holo` codec produces *holographic chunks*: individually useful contributions that can be recombined from almost any subset.

`holo.net` moves those chunks across harsh UDP links. It frames, segments, reassembles, and keeps chunk identity separate from socket endpoints.

The Holographix “field” layer (`holo.field`, with higher-level planning in `holo.net.mesh` and identity helpers in `holo.net.arch`) treats the set of chunks as a shared perceptual substrate that many nodes can read and write, with local reconstruction and policy-driven healing.

A compact view of roles is captured by the mapping below; it is an analogy used as an engineering compass, not as biology as physics:

| Component | Engineering role | Informal mapping |
|---|---|---|
| `codec` | deterministic representation rules (formats, interleaving, versioning) | genotype |
| `field` | best current reconstruction from surviving fragments | phenotype |
| `cortex` | persistence, aging, deduplication, integrity checks | tissue |
| `mesh` | circulation, gossip, opportunistic replication | ecology |
| `arch` | identity and compatibility (`holo://...` → content identifiers) | receptors |
| `transport` | UDP framing, segmentation, reassembly | impulses |

---

## The holographic codec in one equation

Every encoded signal is split into a coarse component and a residual:

```text
residual = original - coarse_up
````

`coarse_up` is the coarse approximation upsampled back to the original resolution/length. The residual carries high-frequency detail. The codec stores the coarse representation plus a permuted, distributed residual across many chunks so that losing chunks reduces detail rather than invalidating the decode.

Decoding is the same physical idea in reverse: reconstruct coarse; allocate residual filled with zeros; write received residual samples into their positions; missing samples remain zero; add residual back to coarse with clipping. With all chunks present, reconstruction is exact or close (depending on the selected model and compression settings). With a subset, reconstruction remains globally coherent and degrades smoothly.

---

## Golden-ratio interleaving (why it exists, what it guarantees)

If you cut the residual into contiguous blocks, you get brittle locality: lose one block and you lose one region or one time segment. Holographix instead spreads residual samples along a deterministic orbit so that every chunk samples the *entire* signal.

Let the flattened residual be a 1‑D array of length `N`. Define the golden ratio:

```text
phi = (1 + sqrt(5)) / 2  ≈ 1.618033...
1/phi = phi - 1          ≈ 0.618033...
```

Choose a discrete rotation step close to `N/phi` and adjust it until it is coprime with `N`:

```text
step ≈ N/phi
gcd(step, N) = 1
```

Then define a full-cycle permutation:

```text
perm[i] = (i * step) mod N
```

If `B` chunks are produced, chunk `b` takes a strided subsequence of this orbit:

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

The practical claim you can test is not mystical: quality should be primarily a function of “how many samples arrived”, not “which chunk IDs arrived”. A good holographic layout yields low variance across random subsets of equal size, and degrades without catastrophic discontinuities.

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
```

The separation is deliberate. The codec does not depend on sockets. The transport does not depend on thumbnails or waveforms. The field logic does not depend on networking primitives. That boundary is what lets you alter mesh policy without touching codec math, and evolve models without rewriting packet transport.

---

## Installation

A recent Python 3 with NumPy and Pillow is sufficient for images.

```bash
git clone https://github.com/ciaoidea/Holographix.io.git
cd Holographix.io

python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install numpy pillow
```

Audio uses the standard library `wave`. Networking uses the standard library `socket` and `struct` plus the modules under `holo.net`.

---

## Quick start (CLI)

Encoding produces a `.holo` directory containing `chunk_XXXX.holo` files.

```bash
# encode an image with default chunk sizing
python3 -m holo image.png

# encode with target chunk size around 32 KB
python3 -m holo image.png 32

# decode from the holographic directory back to an image
python3 -m holo image.png.holo

# audio (PCM WAV)
python3 -m holo track.wav 32
python3 -m holo track.wav.holo
```

To observe graded reconstruction, delete or move some `chunk_*.holo` files and decode again. The output should remain valid and globally coherent, with reduced detail.

---

## Quick start (Python)

```python
import holo

# image
holo.encode_image_holo_dir("image.png", "image.png.holo",
                           target_chunk_kb=32)
holo.decode_image_holo_dir("image.png.holo", "image_recon.png")

# audio
holo.encode_audio_holo_dir("track.wav", "track.wav.holo",
                           target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_recon.wav")
```

---

## Multi-object holographic storage (containers)

When several sensory objects belong to the same conceptual entity, it is often preferable to store them in one holographic field so that loss degrades them *collectively* rather than destroying one file while leaving another perfect. The container module does this by concatenating residual vectors and interleaving them along one long golden trajectory, while carrying the coarse representation for each object as metadata.

```python
import holo

holo.pack_objects_holo_dir(
    ["image1.jpg", "image2.jpg", "track.wav"],
    "pack1.holo",
    target_chunk_kb=32,
)

holo.unpack_object_from_holo_dir("pack1.holo", 0,
                                 output_path="image1_rec.png")
holo.unpack_object_from_holo_dir("pack1.holo", 1,
                                 output_path="image2_rec.png")
holo.unpack_object_from_holo_dir("pack1.holo", 2,
                                 output_path="track_rec.wav")
```

The resulting behaviour is “concept-cloud like”: losing fragments reduces detail across the whole pack, instead of randomly annihilating a single member.

---

## Fields and healing (local metabolism)

A `Field` instance tracks which chunks are present for a given `content_id`, reports coverage, and can decode the best current percept at any time.

```python
from holo.field import Field
from PIL import Image

f = Field(content_id="demo/image", chunk_dir="image.png.holo")

summary = f.coverage()
print("present blocks:", summary["present_blocks"],
      "out of", summary["total_blocks"])

img = f.best_decode_image()
Image.fromarray(img).save("image_best.png")

f.heal_to("image_healed.holo", target_chunk_kb=32)
```

Healing is policy, not magic. It does not recreate missing information. It takes the best currently reconstructable percept, re-encodes it into a fresh holographic population, and restores a clean distribution of coarse and residual data. The purpose is to prevent slow entropic decay when fragments are lost over time and to keep the field usable under long-lived impairment.

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
    send_chunk(sock, addr, content_id, chunk_id, chunk_bytes, max_payload=1200)
```

Above raw transport, `holo.net.mesh` adds gossip about which content IDs exist where and decides what to replicate and repeat. The intended style is that mesh policy remains small and explicit so different agents can adopt different replication strategies while reusing the same codec and framing.

A practical note for harsh links: UDP segmentation turns one logical chunk into many datagrams. On lossy links, “receive the entire chunk” can become significantly less likely than “receive most datagrams”. A field-centric evolution path is therefore to make the smallest network contribution coincide with the smallest decodable contribution, so partial arrivals still improve the percept. The repository keeps codec and transport separate precisely to allow that evolution without entangling math and sockets.

---

## Measuring resilience (turning intuition into curves)

A holographic layout is not a vibe; it is measurable. Fix an input signal, encode into `B` chunks, then for each `k` in `[1..B]` draw many random subsets of size `k`, decode, and measure quality against the original. For images, PSNR/MSE are a reasonable first pass. For audio, SNR is a baseline and perceptual measures can be added if needed.

Two expected signatures indicate genuine interchangeability: mean quality improves smoothly with `k`, and variance across subsets at fixed `k` stays small. When those hold, quality depends mostly on how many fragments survived rather than on which specific identifiers survived.

If you care about interaction realism (prosody, facial motion, affect), it is also worth measuring reconstruction stability as fragments arrive in time with burst loss and reordering. The goal is not only “good after enough data”, but “continuous without spurious discontinuities during acquisition”.

---

## Conceptual lineage (kept explicit and testable)

Holographix borrows language from biology—morphogenesis, fields, healing—because it describes a distributed pattern that remains recognisable under constant material loss. The implementation stays strictly within explicit data structures and deterministic reconstruction rules; no non-material mechanism is assumed.

The use of golden-ratio steps is an engineering technique for near-uniform sampling under modular rotation, chosen to spread residual detail globally with low bookkeeping.

The project also adopts a methodological stance: the deepest design work happens at the level of chosen concepts and axioms (representation as fields, identity addressing, graded reconstruction) and is then tested by concrete experiments on impaired networks.

---

## References

A. Rizzo, *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. DOI: 10.4236/apm.2023.139038



