# <img width="36" height="36" alt="HolographiX logo" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX
## V3.0 — Information fields for lossy, unordered worlds

| <img width="20" height="20" alt="paper" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464) | <img width="20" height="20" alt="book" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="github" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source](https://github.com/ciaoidea/HolographiX) | <img width="20" height="20" alt="medium" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/the-best-so-far-economy-why-i-m-betting-on-fields-not-streams-093b176be1e8) | <img width="20" height="20" alt="podcast" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast [2025 Dec 17th](https://github.com/user-attachments/assets/a3b973a8-d046-4bea-8516-bd8494601437) |

<img width="1280" alt="holographix cover" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

## What this repo is (the point, upfront)

HolographiX implements a **holographic transport layer for information**.

In optical holography, you can cut the plate and you still see the whole scene: the fragment does not “contain a piece of the image”, it contains a *global encoding* of the scene, with reduced detail. This repository takes that property out of optics and turns it into a general engineering primitive: **encode information into interchangeable chunks such that any non-empty subset decodes to a coherent best‑so‑far estimate, improving smoothly as more chunks arrive**, even under loss, reordering, duplication, delay, and partial availability.

That is the generalization: *holography as a software layer*, applicable wherever information is moved, stored, fused, or computed. Networks are one use-case; they are not the definition.

## Why “transport layer”, not “codec marketing”

Traditional stacks push fragility upward: UDP gives you loss and reordering; applications patch over it with retries, buffering, strict ordering, or they give up and tolerate holes. HolographiX flips the responsibility: the representation is built for hostile transport, so **transport unreliability becomes a graceful quality knob**, not a failure mode.

You can drop the same chunk set onto UDP meshes, filesystems, object stores, DTN links, audio/radio modems, or direct inference pipelines. The chunks stay the same; the medium changes.

## Formal contract (what the layer guarantees)

Let x be the source signal and C = E(x) = {c_1,...,c_N} the chunk set.

For any subset S ⊆ C, S != ∅, decoding returns:

x_hat(S) = D(S)

The design target is operational:

x_hat(S) must be coherent for every non-empty S, and as |S| grows the reconstruction must refine without introducing geometric holes (missing pixels, missing time spans). Missing evidence expresses as loss of detail/precision, not missing coordinates.

This property is the “holographic” part: each chunk touches the whole support of the signal.

## What this changes for communication networks (the re-alphabetization)

Once information is a field rather than a stream, the network’s job stops being “deliver packet i after packet i-1” and becomes “diffuse evidence”. Inventory/want, prioritization, partial decoding, healing, stacking, and recovery chunks become first-class operations because they act on fields. You can treat a lossy mesh like a soft channel that gradually increases SNR at the receiver rather than a binary success/failure pipe.

In that sense this is not a protocol tweak; it is a new unit of transport: not packets of ordered stream, but *holographic chunks of a field*.

## Why this also points at neural networks

Neural systems are information channels too. Weights, activations, gradients, and tokens are carriers of evidence. If you can represent those carriers as **best‑so‑far fields**, you can run “anytime” inference and asynchronous fusion under partial availability: compute on partial reconstructions, then refine as more evidence arrives, without waiting for full completion or strict ordering.

This repository focuses on image/audio first because “best‑so‑far” is measurable (PSNR, MSE) and visible/audible. The layer is meant to generalize beyond media.

## Mechanism in one paragraph

HolographiX encodes **coarse + residual**. The coarse part gives an immediately-decodable scaffold; the residual carries refinement. A deterministic **golden-ratio permutation** (the MatriX / golden interleave) spreads residual evidence across chunks so that *which* chunks you have matters far less than *how many* chunks you have. Decoding is stateless and deterministic: any subset yields a best‑so‑far, and adding chunks refines it.

<p align="center">
  <img width="1280" alt="graded reconstruction" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" /><br/>
  <em>Graded reconstruction: fewer chunks soften detail without holes.</em>
</p>

## v3.0 (olonomic) in one sentence

v3 moves residuals into **local wave bases** (DCT for images, STFT for audio), so missing chunks correspond to missing coefficients (“missing waves”), which shrinks chunk sizes and preserves graceful degradation.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src
# or run in-place: PYTHONPATH=src python3 -m holo ...
````

## Quick start

Note: `.holo` output directories (like `src/flower.jpg.holo/`) are generated locally and are not committed to the repo. Run the encode step before any example that reads `*.holo`.

Encode / decode an image (v2, pixel residuals):

```bash
python3 -m holo src/flower.jpg 32
python3 -m holo src/flower.jpg.holo --output out.png
```

Encode / decode olonomic (v3, wave residuals):

```bash
python3 -m holo --olonomic src/flower.jpg --blocks 16 --quality 40
python3 -m holo src/flower.jpg.holo --output out.png
```

Audio:

```bash
python3 -m holo /path/to/track.wav 16
python3 -m holo --olonomic /path/to/track.wav 16
python3 -m holo /path/to/track.wav.holo --output track_recon.wav
```

Packet-sized chunks (mesh/UDP):

```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16
```

## Layering map (codec -> transport -> modem)

```
holo.codec   -> chunk bytes (field representation)
holo.net     -> datagrams (framing + mesh)
holo.tnc     -> audio/radio modem (AFSK/FSK/PSK/etc)
holo.tv      -> multi-frame scheduling (HoloTV windows)
```

<p align="center">
  <img width="800" alt="architecture map" src="https://github.com/user-attachments/assets/ca097bb5-3aaa-4efa-ba5b-8e6495cbae44" /><br/>
  <em>Codec → transport → field layering: genotype/phenotype/cortex/mesh analogy.</em>
</p>

## CLI overview (practical)

The codec CLI is `python3 -m holo`. Decoding auto-detects v2/v3 by chunk directory metadata.

| Intent                  | Example                                                     |
| ----------------------- | ----------------------------------------------------------- |
| Encode to `.holo`       | `python3 -m holo INPUT [TARGET_KB]`                         |
| Decode (all chunks)     | `python3 -m holo INPUT.holo --output out`                   |
| Decode anytime (best‑K) | `python3 -m holo INPUT.holo --max-chunks K --prefer-gain`   |
| v3 encode               | `python3 -m holo --olonomic INPUT --blocks N --quality Q`   |
| Add recovery chunks     | `python3 -m holo --recovery rlnc --overhead 0.25 ...`       |
| Decode with recovery    | `python3 -m holo INPUT.holo --use-recovery`                 |
| Write uncertainty       | `python3 -m holo INPUT.holo --write-uncertainty`            |
| Heal a field            | `python3 -m holo INPUT.holo --heal` or `--heal-fixed-point` |
| Stack exposures         | `python3 -m holo --stack dir1 dir2 ... --output out.png`    |

Help/navigation:

```bash
python3 -m holo --help
python3 -m holo net --help
python3 -m holo tnc-tx --help
python3 -m holo tnc-rx --help
python3 -m holo tnc-wav-fix --help
```

## Python API highlights

```python
import holo
from holo.codec import (
    encode_image_holo_dir, decode_image_holo_dir,
    encode_audio_holo_dir, decode_audio_holo_dir,
    encode_image_olonomic_holo_dir, decode_image_olonomic_holo_dir,
    encode_audio_olonomic_holo_dir, decode_audio_olonomic_holo_dir,
    stack_image_holo_dirs,
)

encode_image_olonomic_holo_dir("frame.png", "frame.holo", block_count=16, quality=60)
decode_image_holo_dir("frame.holo", "frame_recon.png")   # auto-dispatch by version

encode_audio_olonomic_holo_dir("track.wav", "track.holo", block_count=12, n_fft=256)
decode_audio_holo_dir("track.holo", "track_recon.wav")
```

## Field operations (why the “layer” is more than encode/decode)

Recovery (RLNC) adds `recovery_*.holo` chunks to reconstruct missing residual slices under heavy loss. Healing re-encodes the current best‑so‑far into a clean distribution of chunks. Uncertainty outputs quantify what was observed vs inferred. Chunk scoring lets you prioritize transmission and decode “best‑K” subsets.

This is the operational toolset you want when the primitive is a field.

<p align="center">
  <img width="1280"  alt="photon collector" src="https://github.com/user-attachments/assets/c2b939d1-8911-4381-8bd7-a93e29f5401c" /><br/>
  <em>Photon-collector stacking: multiple exposures reinforce structure over noise.</em>
</p>

<p align="center">
  <img width="1280"  alt="psnr curves" src="https://github.com/user-attachments/assets/e8c700f2-e5b6-424b-a848-a230294e8269" /><br/>
  <em>PSNR vs received chunks: quality rises smoothly; variance stays low.</em>
</p>

## Olonomic v3 (operational sketch)

Images (v3) encode the residual in a block DCT domain:

img -> coarse(img) -> coarse_up
residual = img - coarse_up
residual -> pad to block size -> DCT-II per block/channel
coeffs -> quantization (quality 1..100) -> zigzag
coeff stream -> golden permutation -> slice across chunks -> zlib per chunk
decode: inverse steps + coarse + residual

Audio (v3) encodes the residual in an STFT domain:

audio -> coarse(audio) -> coarse_up
residual = audio - coarse_up
residual -> STFT (sqrt-Hann, hop ~ n_fft/2 default)
bins -> frequency-shaped quant -> int16 (Re/Im)
stream -> golden permutation -> chunks -> zlib
decode: ISTFT overlap-add + coarse + residual

Metadata is packed inside coarse payloads so chunk headers stay backward-compatible.

## Repository map

```
README.md                  top-level overview (this file)
src/pyproject.toml         packaging for editable install
src/requirements.txt       runtime deps (numpy, pillow)

src/holo/
  codec.py                 codecs v1/v2/v3 (image/audio), chunk scoring, recovery hooks
  recovery.py              GF(256) RLNC recovery chunks + solver
  __main__.py              CLI entry (codec + tnc-tx/tnc-rx)
  container.py             multi-object packing/unpacking
  field.py                 field tracking + healing (fixed-point)
  cortex/                  storage helpers
  net/                     transport + mesh + content IDs
  models/                  coarse model abstraction
  tnc/                     modem + framing + WAV CLI
  tv/                      HoloTV scheduling + demux helpers

src/examples/              runnable demos
src/tests/                 unit tests
src/docs/                  Global Holographic Network guide
src/infra/                 containerlab + netem/benchmark configs
```

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -p 'test_*.py'
```

Network/mesh smoke test (loopback UDP using `holo://` content IDs):

```bash
PYTHONPATH=src python3 src/examples/mesh_loopback.py
```

## Results snapshot

On `src/galaxy.jpg`, `coarse-side=16`, v3:

`python3 -m holo --olonomic src/galaxy.jpg --blocks 16 --quality 40 --packet-bytes 0`

Total is ~0.35 MB (coherent even from a single chunk). Same settings in v2 (pixel residuals) are ~1.69 MB. v3 degrades mainly as missing waves, not missing pixels.

## References

Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. [https://doi.org/10.1016/0001-6918(86)90062-4](https://doi.org/10.1016/0001-6918%2886%2990062-4)

Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4

Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).

Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9

Rizzo, A. (2023). *The Golden Ratio Theorem*. Applied Mathematics, 14(09). [https://doi.org/10.4236/apm.2023.139038](https://doi.org/10.4236/apm.2023.139038)

Rizzo, A. (2025). *HolographiX: Holographic Information MatriX for Resilient Content Diffusion in Networks* (v1.6.2). [https://doi.org/10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464)

Rizzo, A. (2025). *HolographiX: From Fragile Streams to Information Fields*. ISBN-13: 979-8278598534. [https://www.amazon.com/dp/B0G6VQ3PWD](https://www.amazon.com/dp/B0G6VQ3PWD)

<p align="center">
  © 2025 <a href="https://holographix.io">holographix.io</a>
</p>

