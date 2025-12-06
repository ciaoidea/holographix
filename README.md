# Holographix

Holographic media and networking for resilient, extreme connectivity

Holographix (from "holographic information matrix") is a resilient, awareness-oriented network built on top of a holographic codec and a UDP transport. It is meant as a perceptual substrate for large sensory models — LVMs (Large Vision Models) and LVAs (Large Audio Models) — running over networks that behave like reality, not like a clean lab: loss, jitter, reordering, duplication, fading links, mobility.

At the lowest layer, the `holo` codec turns images and audio into holographic chunks that can be recombined from almost any subset. On top of that, `holo.net` pushes those chunks across harsh UDP links. Above both, the Holographix layer (`holo.field`, `holo.cortex`, `holo.net.mesh`, `holo.net.arch`) treats chunk clouds as shared perceptual fields that many nodes can read from and write to.

The core question is: how do you keep useful information alive when the network is temporary, damaged or improvised, and when the classic idea of a reliable stream between two fixed endpoints is more fiction than reality?

The aim is not to deliver every bit. The aim is to maximise the quality of the percept per bit that actually survives, under changing loss, latency, noise and mobility. The system is built to adapt to the environment in the same way living tissue does: by distributing structure, not by pretending the medium is perfect.

Current scope: RGB images and PCM WAV audio. Generic opaque binaries are deliberately out of focus for now, because they usually need strict erasure coding rather than graceful blur.

---

## Origin: life, radios and the shape of resilience

Holographix did not start from networking textbooks; it started from a very old human experience. Lying in bed at three years old, listening to Neil Diamond's "Song Sung Blue" on a bedside radio, the memory that survived was not a waveform. It was a pattern of melody, rhythm and emotion that stayed intact through decades of noise, forgetting and change. Biology does something remarkable here: it keeps meaning alive even when almost all physical detail has been lost.

Years later, in the basement of a country house in Villetta Barrea near Rome, a dusty shortwave tube radio brought that intuition into hardware. The set was older than I was; when its tubes lit up it pulled voices out of a noisy, unstable ether. The signal faded, warped, picked up static and drifted in frequency, yet the message remained accessible because the receiver and the brain behind it were built to infer from partial evidence, not to demand perfect samples.

A CB radio then opened the door to amateur radio, a callsign (IK2TYW), and eventually physics. On HF and VHF, working the MIR space station and Russian polar satellites such as RS-10, RS-11 and RS-12, the contrast became very sharp. Digital protocols on clean cables expect a perfect packet or nothing, and they block on a single missing fragment. Analog links and human perception do the opposite: they accept that reality is broken and move on, degrading gracefully instead of collapsing.

The design of Holographix takes that pattern as a specification. The resilience we want from the codec and from the network is not a generic robustness metric; it is the same structure that keeps a living system recognisable while its material is constantly replaced. A song in memory, a face through rain on a window, a voice across static: the form that survives is the life of the information. The project tries to encode that form into bytes and UDP packets.

---

## Perception, language and emotional AI

Modern neural systems do not think in a single way. Large language models work in a discrete, symbolic regime. They process sequences of tokens where a misplaced or missing element can flip a truth value, break a program, corrupt a JSON document or destroy a chain of reasoning. Their inference is sensitive to discrete jumps: a single error can push the system into a completely different state. For this kind of object, there is no clean notion of "slightly wrong"; syntactic or logical validity is often all-or-nothing.

Vision and audio models live in a different regime. They act on continuous fields with strong redundancy: spatial structure, temporal continuity, spectral smoothness, motion constraints, cross-modal correlations. When evidence is missing, they typically lose resolution or confidence, but the scene, the gesture or the phrase often remain recognisable. A face can be detected under rain; a word can be understood through noise; a rhythm can be followed even if half the samples are gone. This is not a miracle; it is the signature of a field-like representation where information about the whole is spread across many parts.

Holographix is deliberately tuned to this sensory regime. It is designed as a substrate where LVMs and LVAs can keep working in an anytime fashion. At any moment, the `Field` layer can offer a "best current percept" reconstructed from whatever chunks are present. If more fragments arrive later, reconstruction improves, but the models do not need to stop and wait for perfection. This is exactly the kind of behaviour that makes emotional interaction possible. A system that estimates affect from prosody, gesture, timing or facial motion does not need bit-perfect data; it needs continuity of experience. Hard stalls and file-level corruption introduce discontinuities that feel emotionally wrong. Smooth degradation of audio and video keeps the prosodic and visual contours intact enough that an "emotional AI" can track what matters without hallucinating sudden jumps that are artefacts of the transport.

The psychological and emotional resilience of the network is therefore not a marketing slogan. It is an architectural constraint: preserve the continuity of perceptual fields under loss, because that is where meaning and emotion live for humans and for any AI that tries to share a world with them.

---

## Fields, morphogenesis and morphic intuition

In developmental biology, the term "morphogenetic field" has a precise and non-mystical meaning. It describes the distributed patterning influences that guide the formation of tissues and organs. No single cell contains the plan for a limb; the pattern emerges from gradients, signals and constraints spread out in space and time. If parts are removed, the field can often reconfigure and still produce a recognisable structure. The key idea is that form is stored in a field of relations, not in a single point.

Rupert Sheldrake pushed this intuition much further with his notion of "morphic fields" and "morphic resonance": immaterial fields that would connect patterns of form and behaviour across individuals and history. That proposal is controversial and lies well outside standard physics and biology. Holographix does not rely on any non-material influence; it stays firmly inside explicit data structures, code and measurable behaviour.

What survives from that family of ideas is the picture of form as an attractor sustained by a population. Holographix treats each piece of perceptual content as a field in exactly that engineering sense. A chunk is not a local brick that "owns" a region of the image or of the sound. It is a sample of a global orbit through the residual field. The total population of chunks, plus the deterministic reconstruction rule, define a state that tends to come back to the same percept even when many pieces are missing. When content is re-encoded, replicated and healed, the network is effectively making that percept easier to re-evoke from fragments. That is a very mundane, falsifiable form of resonance, but it carries the same intuitive flavour: patterns that have been reinforced by use are the ones that reappear quickly when the world is incomplete.

In this sense, the holographic codec and the mesh layer implement a digital, testable version of a morphic field: a structure whose essence is to keep meaning alive by distributing it and by letting partial traces cooperate, instead of relying on a single fragile copy.

---

## The golden rule: why interleaving is golden

When you look at how living structures occupy space, you repeatedly encounter arrangements that are neither perfectly regular nor obviously aligned: spirals in plants, seed patterns, branching angles. The interest is not in mysticism or numerology; the interest is in the fact that certain proportions make it easy to fill space without privileging directions and without creating short cycles. The golden ratio is the simplest of these proportions.

Holographix uses the golden ratio to decide how the fine structure of a signal is threaded into chunks. The idea is to flatten the residual into a one-dimensional line and then walk that line with a step that is maximally "incommensurate" with the length. Each step jumps to a new position; the orbit visits every index once before repeating. If you then cut that orbit into several interleaved strands, each strand becomes a representative sample of the whole. Each chunk is a phase slice of that orbit.

Formally, the golden ratio is introduced from the simplest definition. Consider a whole segment and split it into a larger part and a smaller part. The golden condition is

```text
whole : larger  =  larger : smaller
````

If the whole has length 1, the larger part has length x and the smaller part has length (1 − x), the condition becomes

```text
1 / x = x / (1 − x)
x^2 + x − 1 = 0
x = (sqrt(5) − 1) / 2  ≈  0.618033...
```

The classical golden ratio is

```text
phi = (1 + sqrt(5)) / 2  ≈  1.618033...
phi − 1 = 1 / phi  ≈  0.618033...
```

Once the residual is flattened into a one-dimensional array of length `N`, the codec turns the golden fraction into a discrete rotation step:

```text
step ≈ (phi − 1) * N   (i.e. N / phi)
```

The step is then adjusted so that

```text
gcd(step, N) = 1
```

which guarantees a single full cycle. The permutation is

```text
perm[i] = (i * step) mod N
```

If the residual is split into `B` chunks, chunk `b` takes the subsequence

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

Every chunk therefore samples the residual line in a quasi-uniform way instead of cutting out a contiguous block.

Intuitively, this is where the connection to living resilience becomes sharp. The golden orbit is the skeleton of a digital morphogenetic field. No chunk owns a local piece of anatomy; every chunk participates in the same global spiral. When some chunks are lost, the effect is like damaging tissue: the organism becomes less detailed, but its body plan persists. When chunks are regenerated and recirculated, the same form tends to re-emerge, because the way information is spread makes it hard to kill the pattern with a small number of blows.

The golden interleaving is therefore not a decorative choice. It is the concrete structure that makes "graceful degradation" the default behaviour of the codec.

---

## How the holographic codec works in practice

For both images and audio, the codec follows a simple physical pattern.

First it builds a coarse approximation of the signal. For an image, this is typically a small thumbnail that is then resized back to the original resolution. For audio, it is a subsampled version of the track, interpolated back to full length. This coarse layer captures the global structure: geometry and colour layout for images, envelope and slow variations for audio.

Then, in signed integer space, it computes a residual:

```text
residual = original - coarse_up
```

This residual carries the fine detail that is missing from the coarse view. It is flattened into a one-dimensional array, threaded along the golden permutation described above, and split across a chosen number of chunks. Each chunk carries the coarse representation plus its share of the permuted residual.

Decoding reverses the process. The decoder reads any chunk that is available, reconstructs the coarse, allocates a residual array filled with zeros, regenerates the same golden permutation, writes received residual samples into their positions, leaves missing samples at zero, reshapes back to the original geometry and adds the residual to the coarse approximation with clipping to the valid range. When all chunks are present, the reconstruction is exact or very close, depending on the model and on compression. When only a subset is present, the reconstruction is globally coherent but blurrier or more lo-fi. The important part is that there is no catastrophic break: format decoders still see valid images and audio, and perceptual models still see recognisable scenes and phrases.

---

## Architecture and package layout

The repository is organised as a small Python package centred on the `holo` namespace. The goal is to keep responsibilities separated: codec math, field logic, storage and network transport each live in their own module.

The core tree looks like this:

```text
holo/
  __init__.py        public API for image/audio encode-decode,
                     multi-object packing and Field

  __main__.py        command line entry point,
                     argument parsing and dispatch only

  codec.py           single-signal codec:
                     chunk formats, headers, versioning,
                     compression and golden interleaving

  container.py       multi-object packing:
                     one holographic store can contain many objects,
                     each with its own coarse layer

  field.py           local field representation for one content_id:
                     ingest chunks, track coverage, decode best view,
                     perform healing

  cortex/
    __init__.py      helpers for local storage
    store.py         persistent storage backend for chunk sets
    visual.py        convenience helpers for visual experiments

  models/
    __init__.py      small registry that picks the right model
    image.py         image model: coarse thumbnail + int16 residual
    audio.py         audio model: coarse subsampling + int16 residual

  net/
    __init__.py      namespace for networking
    transport.py     UDP framing, segmentation and reassembly
    arch.py          helpers for holo:// URIs and content identifiers
    mesh.py          peer overlay, gossip and chunk replication
```

The codec does not know about UDP or fields. The transport does not know about thumbnails or residuals. The field logic does not know about sockets. This is deliberate: Holographix is meant as a set of clean layers, not as one tangled script.

---

## Installation

You need a recent Python 3 interpreter together with NumPy and Pillow.

```bash
git clone https://github.com/ciaoidea/Holographix.io.git
cd Holographix.io

python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\\Scripts\\activate

pip install numpy pillow
```

Audio support relies on the standard library `wave` module. Networking uses only the Python standard library for sockets and struct packing together with the code in `holo.net`.

---

## Quick start with the codec

The simplest way to use the codec is to work with a single image or audio file and its corresponding `.holo` directory.

From the command line:

```bash
# encode an image into holographic chunks with default sizing
python3 -m holo image.png

# encode with target chunk size around 32 KB
python3 -m holo image.png 32

# decode from the holographic directory back to an image
python3 -m holo image.png.holo

# encode and decode audio (PCM WAV)
python3 -m holo track.wav 32
python3 -m holo track.wav.holo
```

After encoding you will find a directory such as `image.png.holo` or `track.wav.holo` containing `chunk_XXXX.holo` files. Removing some of these chunks and decoding again lets you observe the graded reconstruction behaviour: fewer chunks give a blurrier image or a more noisy audio track, but the result remains globally coherent.

As a Python module:

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

## Multi-object holographic storage

When several images and audio tracks belong to the same conceptual object, it is often more natural to store them in a single holographic field rather than in separate directories. The container module implements this by keeping the coarse representation of each object as if it were encoded alone, concatenating all residual vectors, and interleaving them along one long golden trajectory. Each chunk then carries the complete set of coarse bytes and a different slice of the combined residual.

With the Python API:

```python
import holo

# pack several objects into one holographic field
holo.pack_objects_holo_dir(
    ["image1.jpg", "image2.jpg", "track.wav"],
    "pack1.holo",
    target_chunk_kb=32,
)

# later, reconstruct individual objects by index
holo.unpack_object_from_holo_dir("pack1.holo", 0,
                                 output_path="image1_rec.png")
holo.unpack_object_from_holo_dir("pack1.holo", 1,
                                 output_path="image2_rec.png")
holo.unpack_object_from_holo_dir("pack1.holo", 2,
                                 output_path="track_rec.wav")
```

When chunks are missing, all objects degrade slightly but remain coherent. The field behaves like a small concept cloud: losing fragments reduces detail across the whole cloud, not by randomly erasing one object while leaving another perfect.

---

## Fields and healing

A `Field` instance is the local "metabolism" around one holographic directory. It knows which chunks are present, can estimate coverage and can decode the best view it can produce from the current subset. This is the place where the "life" of the information actually plays out.

Example usage:

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

Healing does not recreate lost information. It takes the best available reconstruction, re-encodes it into a fresh holographic directory, and restores a clean distribution of coarse and residual data. This prevents slow entropic decay when chunks are progressively lost and ensures that the field stays usable over long time scales.

---

## UDP transport and mesh

The UDP transport in `holo.net.transport` is intentionally minimal. It knows about three things: a content identifier, a chunk identifier and how to split and reassemble chunks into UDP packets. It does not know about images, audio or containers; it treats all chunks as opaque byte strings.

A very simple sending loop looks like this:

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

On the receiving side you can feed incoming packets into a `Reassembler` and write complete chunks out to a directory. Above this raw transport, the mesh layer in `holo.net.mesh` adds simple gossip about which content IDs are present where, and decides which chunks to repeat under which conditions. The design goal is to keep policies small and explicit so that different robots or agents can adopt different replication strategies while reusing the same codec and packet format.

The important qualitative difference compared to plain TCP and UDP can be stated in words. A TCP stream gives you strong guarantees about ordering and completeness but can stall or collapse entirely on bad links. A bare UDP socket gives you raw datagrams and no guarantees at all; robustness is entirely your problem. Holographix over UDP pushes robustness into the representation itself. Every chunk is a valid contribution to a perceptual field, and any subset of chunks produces a globally consistent reconstruction whose quality grows smoothly with the number of fragments received.

---

## Resilience experiments

The behaviour of a holographic layout can be measured in a straightforward way. Pick an image, encode it into a fixed number of chunks, and for each possible number of surviving chunks draw many random subsets, reconstruct and measure mean squared error or PSNR against the original. A good holographic layout shows two features: the mean quality improves smoothly with the number of chunks, and the variance across subsets with the same size stays very small. That means chunks are genuinely interchangeable and quality depends mostly on how many fragments survived, not on which specific ones.

The same strategy applies to audio with signal-to-noise ratio or perceptually inspired measures. These experiments turn the intuitive picture of "living tissue that degrades gracefully" into quantitative curves you can compare between codec variants.

---

## Einstein's diagram and where Holographix lives

In a letter to his friend Maurice Solovine dated May 7th, 1952, Albert Einstein drew a small diagram to answer the question "What is science?". Along the bottom he placed a horizontal line marked E, for immediate experiences, the empirical basis. Higher up he placed another line marked A, for the axioms and basic concepts of a theory. From A down to E he drew slanting lines labelled S1, S2 and so on, representing particular statements and predictions deduced from the axioms and tested against experience.

The important claim in that letter is that there is no logical machine that takes you from E up to A. The passage from raw experience to axioms requires an intuitive, extra-logical, psychological leap. Once you have chosen your axioms, you can derive statements, compare them with observations and keep or discard your theory. But the real game is played at the level of A, where you decide which concepts you are going to use to see the world. For that reason Einstein did not draw a sharp boundary between science and philosophy. Both live in the space of possible conceptual schemes.

Holographix is very explicit about the fact that it is a choice at that level. At the E level we have packets on the wire, bit errors, microphone samples, camera pixels, robot positions. The project does not pretend to deduce its main ideas from these raw data. Instead it chooses a small set of axioms: that perceptual content should be represented as fields rather than as brittle objects, that resilience of meaning is not an afterthought but the primary structure, that a golden interleaving is a good way to spread fine detail, that content should be addressed by identity rather than by host location, and that psychological and emotional coherence in interactions is a legitimate design target.

From these axioms follow all the more concrete statements of the system: holographic chunks, graded reconstruction, healing, mesh behaviour, and the way LVMs and LVAs can sit on top of `Field` to keep updating their beliefs under loss. These statements can be tested in practice on radios, robots and unstable networks. The diagram is a reminder that the deepest part of the design is not the code that does interpolation or the specific headers on the wire, but the decision to treat information as a living field whose purpose is to keep meaning alive.

---

## References and lineage

The mathematical use of the golden ratio in the sampling of residuals is informed by work such as A. Rizzo, "The Golden Ratio Theorem", Applied Mathematics, 14(09), 2023, which shows how golden-ratio steps yield near-uniform coverage when sampling.

The use of the language of morphogenetic and morphic fields is inspired by classical developmental biology and by Rupert Sheldrake's proposals on morphic resonance, but the implementation here stays strictly within standard digital communication and signal processing.

The reflection on where this kind of design lives in the space of ideas follows the spirit of Einstein's 1952 letter to Maurice Solovine on the relation between experience, axioms and scientific concepts.
