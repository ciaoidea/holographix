# Holographix

**Holographix (from “holographic information matrix”) is a percept-first codec and network substrate for running LVM/LVA over UDP.**  
Here “LVM” means Large Vision Model, “LVA” means Large Audio Model: systems that should keep seeing and hearing even when the channel behaves like reality (loss, jitter, reordering, duplication, mobility). Holographix does not try to preserve *all bits* at all costs. It tries to preserve *meaning* under failure, by making every packet carry a usable, global hint of the whole.

A normal file transfer is brittle: it preserves bits or collapses. Holographix flips the goal. Images and audio are handled as **living fields**: a population of fragments circulates, any node reconstructs the best current percept from whatever survives, and quality improves smoothly as more fragments arrive. UDP is the right failure mode for this: it does not pause the world waiting for perfect order; it lets the system estimate now, refine later.

This repository currently targets **RGB images** and **PCM WAV audio**. General binary data is intentionally out of scope for now, because it typically needs strict erasure coding rather than perceptual degradation.

---

## Inspiration / Philosophy (the constraint, not the poetry)

Holographix was born from a very human place: a childhood memory of lying in bed at 3 years old, listening to Neil Diamond’s *Song Sung Blue* on a bedside radio. That memory survived for decades not as a perfect file, but as a robust, emotional imprint. Much later, the question became unavoidable:

> Why does the brain keep a song through half a century of noise and decay, while a digital file can die because one packet is wrong?

The question re-emerged in the basement of a country house in Villetta Barrea, near Rome. In the home of Franco—my father’s best friend, like a second father to me, and uncle to Paolo and Patrizia, who felt like siblings—I found a dusty shortwave tube radio, older than I was. When those tubes lit up, it wasn’t just a machine coming alive. It was a living receiver that could still pull voices out of the ether fifty years later.

Around the same time, my best friend Ivan and his father’s CB radio pulled me into the world of radio. That path led to a ham radio license (IK2TYW) and to physics. On the air I didn’t just talk across town: I worked the MIR space station and tracked Russian polar satellites (RS-10, RS-11, RS-12). Those weak, drifting signals taught a sharp contrast. In the digital world (TCP/IP), a damaged packet is dropped and the connection stalls. In the analog world (radio/physics), when MIR tumbled or the ionosphere cracked, the signal faded, distorted, picked up static — but the message was still there. The human brain, the ultimate holographic decoder, could reconstruct meaning from partial information.

In the 1990s I watched two networks grow side by side. On one side the wired Internet was exploding: Ethernet, routers, TCP/IP, perfect packets or nothing. On the other, ham packet radio was quietly doing its job: slow, noisy, full of RF interference, yet small messages eventually arrived, byte by byte, over HF and VHF links. One network assumed clean cables and punished every error; the other accepted noise as a fact of life and let information seep through anyway. In parallel, biology was showing yet another kind of network: nervous systems, mycelium, vascular systems, all communicating through imperfect, redundant signals that converge rather than fail on a single glitch.

Holographix exists to bring that analog and biological resilience into the digital domain, specifically where perception must not stop.

---

## Morphogenetic fields, “morphic fields”, and what Holographix actually claims

The word *field* is doing real work in this project, but not in a supernatural sense.

In mainstream developmental biology, a *morphogenetic field* is a way to name something very concrete: patterning control that is distributed across space and time. A structure is not “stored in a single cell”; it emerges from many coupled signals, and it can often regulate and recover when parts are removed. In one rigorous phrasing, the morphogenetic field can be defined as the integrated set of non-local patterning signals impinging on cells over 3D space and time. :contentReference[oaicite:0]{index=0}

Rupert Sheldrake pushed a more radical extension. Building on the language of morphogenetic fields, he proposed *formative causation* and “morphic resonance”: a spatio‑temporal influence whereby the more often a self‑organizing pattern occurs, the easier it tends to occur again. :contentReference[oaicite:1]{index=1} This is controversial, and the goal of this repository is not to validate non-local memory or paranormal channels. The engineering stance here is stricter: **everything must reduce to explicit data structures, codecs, transports, and testable behavior.**

So why mention Sheldrake at all?

Because the intuition is exactly the one we implement in software, without metaphysics: **form as an attractor sustained by distributed constraints.** Holographix makes “the form” (an image, a sound) persist as a *field state* in the Field layer: an evolving reconstruction driven by whatever fragments exist right now. A chunk is not a local slice; it is a constraint that touches the whole. As fragments circulate and regenerate, the network increases the probability that the same coherent percept reappears quickly from partial evidence. That is “resonance” in an entirely material sense: replication, mixing, and healing turn a fragile file into a stable attractor under loss.

If you like the language of morphic fields, you can read Holographix as a deliberately grounded, falsifiable version of that metaphor: the “field” is the distribution of fragments plus the reconstruction policy, and its “memory” is implemented as storage, circulation, and regeneration—not as an immaterial entity.

---

## Psychological and emotional resilience (as a design target, not a slogan)

The childhood-radio origin story matters because it points to a specific property of human memory: it is not bit-perfect, and it is not ashamed of that. What survives is a *coherent gist* plus the *emotional contour*—the aspects that remain meaningful when detail is missing.

Holographix encodes that bias on purpose.

The coarse layer is the “body plan”: global structure in an image, global rhythm/contour in audio. The residual is fine texture: timbre microstructure, edge detail, high-frequency nuance. When the channel is harsh, what you lose first is detail; what you keep is the part that still supports perception, recognition, and affect. That is why a degraded reconstruction can remain emotionally intelligible (you still hear the phrase, the prosody, the rhythm; you still see the scene layout, the gesture, the face geometry) instead of collapsing into “file corrupted”.

This is also where “emotional AI” becomes an engineering statement instead of a vibe. Models that infer affect from audio/video (prosody, timing, facial motion, posture, context) are extremely sensitive to discontinuities: dropouts and stalls create false spikes, broken temporal cues, and unstable embeddings. A percept-first UDP substrate that degrades smoothly does something very specific for these systems: it preserves continuity of sensory evidence. The result is not that the machine “feels”; it is that the interaction keeps its emotional coherence because the percept does not shatter when the network does.

In short: Holographix treats *meaning* as something that must persist through noise, and it treats the *emotional layer of meaning* as part of the payload that deserves protection via graceful degradation and regeneration, not via brittle correctness.

---


---

## The idea in one picture (without mysticism)

A chunk is not “a piece of the file”. A chunk is closer to a cell: it carries a complete low-resolution “body plan” plus a scattered sample of fine detail. Lose cells and the organism does not become corrupt; it becomes blurry. Under harsh conditions the system can “heal”: it can generate a fresh population of chunks from the best reconstruction it currently has, possibly with different redundancy parameters that fit the channel.

That is what “awareness-oriented” means here: the network is judged by the quality of the percept it keeps alive, not by bit-perfect delivery.

---

## Why this exists for LVM/LVA (and why language behaves differently)

Large language models operate on a **discrete symbolic stream**. Tokens are brittle in the engineering sense: drop or corrupt the wrong symbol and you may destroy syntax, flip a negation, break JSON, change a variable name, or derail a chain of reasoning. Symbolic inference often fails in phase transitions: it looks fine, then suddenly it doesn’t parse, or it becomes logically inconsistent. For language, “graceful degradation” is not a stable objective unless you introduce explicit redundancy at the symbol level.

Audio and video models live in a different regime. They infer from **continuous fields** with strong physical redundancy: temporal continuity, spectral structure, spatial coherence, motion constraints, and cross-sensor correlations. Their internal inference is closer to state estimation than to symbolic parsing. When evidence is missing, they usually do not collapse; they reduce confidence, lose fine detail, or blur boundaries. That is exactly the failure mode you want over UDP: you want “some evidence now” to drive an LVA/LVM forward, and you want refinement when more evidence arrives. This is the key architectural split in the “thought stack” of modern neural systems:

Perception (audio/video) can be designed as an anytime process with partial, improving estimates. Language (symbolic control, tool calls, code, commitments) often needs stricter guarantees and therefore different transport policy. Holographix is built to serve the perception side: it ships **sensory evidence** so that LVM/LVA can keep running under loss, without pretending that symbol streams behave the same way.

---

## How the holographic codec works

For both images and audio the codec follows the same physical pattern: build a coarse percept first, then encode the missing fine structure as a residual, then spread that residual globally so every chunk is “a bit of everything”.

In signed integer space:

```text
residual = original - coarse_up
````

Today the coarse signal is simple and deterministic. For images, the coarse is a thumbnail stored as PNG and then upsampled to full resolution. For audio, the coarse is a subsampled track that is interpolated back to full length. The coarse is not the interesting part; it is the anchor that keeps the reconstruction globally coherent when detail is missing.

The residual is where detail lives. Instead of storing residual contiguously (which would make some chunks carry only local parts), the codec interleaves residual samples along a golden-ratio full-cycle permutation. That makes every chunk statistically similar: each one carries a global coarse view plus scattered detail.

The practical result on a damaged channel is exactly what you want:

If all chunks arrive, reconstruction is exact (or very close, depending on the model).
If only some arrive, you still get a globally coherent result. Missing chunks mainly remove fine detail rather than breaking the format.

---

## The golden rule (why “golden” interleaving)

The codec spreads residual samples across chunks using the simplest definition of the golden ratio:

```text
whole : larger  =  larger : smaller
```

Let the whole be 1, the larger part be x, and the smaller be (1 − x). The condition becomes:

```text
1 / x = x / (1 − x)
x^2 + x − 1 = 0
x = (sqrt(5) − 1) / 2  ≈  0.618033...
```

The classical golden ratio is:

```text
phi = (1 + sqrt(5)) / 2  ≈  1.618033...
phi − 1 = 1 / phi  ≈  0.618033...
```

### From the golden fraction to a full-cycle permutation

Once the residual is flattened into a 1D line of length N, the codec turns the golden fraction into a discrete “rotation step”:

```text
step ≈ (phi − 1) * N   (i.e. N / phi)
```

Then it adjusts step so that gcd(step, N) = 1, guaranteeing a single full cycle. The permutation is:

```text
perm[i] = (i * step) mod N
```

If the residual is split into B chunks, chunk b takes the subsequence:

```text
perm[b], perm[b + B], perm[b + 2B], ...
```

This is why chunk loss produces graceful degradation: every chunk samples the whole residual line in a quasi-uniform way, instead of slicing a contiguous region.

---

## Biology mapping (useful, not poetic)

Holographix is not modeling a brain. The mapping is used as an engineering compass:

```text
codec     = genotype     (chunk format, versioning, golden interleaving rules)
field     = phenotype    (best current reconstruction from surviving chunks)
cortex    = tissue       (local storage, aging, dedup, integrity checks)
mesh      = ecology      (circulation, gossip, opportunistic replication)
arch      = receptors    (identity: holo://... -> content_id, compatibility)
transport = impulses     (UDP framing/segmentation/reassembly)
```

The codec is deterministic and versioned math. Healing is policy.

---

## Repository layout (clean separation of responsibilities)

```text
holo/
  __init__.py
    Stable public API (images + audio):
      encode_image_holo_dir / decode_image_holo_dir
      encode_audio_holo_dir / decode_audio_holo_dir
      container helpers (packing multiple objects)          [work in progress]
      Field, CortexStore                                   [work in progress]
      URI helpers: holo://... -> content_id                [work in progress]

  __main__.py
    CLI entrypoint:
      parses args
      dispatches to public API
      never contains codec math or storage logic

  codec.py
    Genotype layer:
      chunk formats (magic bytes, headers, payload layout)
      versioning
      compression
      golden interleaving of residual arrays

  models/
    __init__.py
      Model registry / common interface (image, audio).

    image.py
      Image model:
        coarse thumbnail PNG
        residual int16(original − coarse_up)

    audio.py
      Audio model:
        coarse subsampling + interpolation
        residual int16(original − coarse_up)
        (future direction: STFT/MDCT coarse models for better psychoacoustics)

  container.py
    Multi-object packing                                   [work in progress]
    Rule: each object keeps its own coarse as if alone,
    residuals are mixed along one long golden trajectory.

  field.py
    Field: live reconstruction for one content_id          [work in progress]
    Ingest chunks, track coverage, decode best percept, trigger healing.

  cortex/
    store.py
      Persistent local storage (tissue)                    [work in progress]

    visual.py
      Visual convenience layer                             [work in progress]

    audio.py
      Audio convenience layer                              [work in progress]

  net/
    transport.py
      UDP framing/segmentation/reassembly                  [work in progress]

    mesh.py
      Peer overlay + gossip + replication                  [work in progress]

    arch.py
      holo://... identity + version negotiation            [work in progress]
```

The design rule is strict: no file mixes CLI parsing, codec math, storage policy and network policy.

---

## Installation

You need Python 3 plus NumPy and Pillow:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy pillow
```

Audio uses the Python standard library wave. Networking uses standard modules (socket, struct, select) when enabled.

---

## Quick start (single image / single audio)

The codec works on a directory of chunks. Encoding produces `something.holo/` containing `chunk_XXXX.holo`. Decoding reads any subset of those chunks.

From Python:

```python
import holo

# Image
holo.encode_image_holo_dir("image.jpg", "image.jpg.holo", target_chunk_kb=32)
holo.decode_image_holo_dir("image.jpg.holo", "image_rec.png")

# Audio (PCM WAV)
holo.encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_rec.wav")
```

To see graceful degradation, delete a few `chunk_*.holo` files from the directory and decode again. You should get a coherent but blurrier image, or a noisier/lo-fi audio track rather than a broken file.

---

## Networking: UDP as perceptual transport for LVM/LVA

UDP is used because it fails the way the physical world fails: packets drop, reorder, duplicate and arrive late. A codec that is intrinsically tolerant to partial information turns UDP from “unreliable” into “gracefully lossy”.

The awareness-oriented behavior appears in how chunks are consumed:

A node does not wait for completeness. It keeps decoding what it has and exposes the best current percept to higher systems.
If more chunks arrive later (from itself or from peers), quality improves smoothly.
The mesh does not ship by file path; it ships by content_id derived from `holo://...`.

Field and Mesh split the responsibility:

Field keeps the current phenotype: it knows coverage and can produce the best decode repeatedly.
Mesh exchanges chunks opportunistically between peers and replicates what matters under observed loss/jitter.

The long-term destination is content addressed by identity (`holo://...`) rather than by host endpoints. A node asks the mesh for fragments belonging to a content_id and reconstructs from the field it accumulates, feeding LVM/LVA with “good enough now, better later” evidence.

---

## Healing: regeneration instead of brittle correctness

Passive robustness is not enough on a long-lived system. If you keep losing chunks, eventually quality decays.

Holographic healing is the active mechanism:

The Field decodes the best current reconstruction (even if degraded), then re-encodes it into a fresh new population of chunks, optionally with updated redundancy/parameters for the environment. The phenotype becomes the seed of a new generation.

This does not recover information that is truly gone. It restores a healthy distribution of what remains, so the system stays stable instead of slowly collapsing.

---

## Experiments

The `resilience_psnr/` folder contains scripts that measure image reconstruction quality (MSE/PSNR) as a function of how many chunks survive. The thing to watch is not only the mean, but the variance: good holographic layouts make chunks interchangeable, so quality depends mostly on “how many” chunks, not “which” chunks.

Analogous experiments for audio can use SNR or perceptual measures (future direction).

---

## Reference

A. Rizzo, *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. [DOI: 10.4236/apm.2023.139038](https://doi.org/10.4236/apm.2023.139038)
