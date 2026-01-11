# <img width="36" height="36" alt="HolographiX logo" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX
## V3.0 — Information fields for lossy, unordered worlds

| <img width="20" height="20" alt="paper" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.18017872](https://doi.org/10.5281/zenodo.18212768) | <img width="20" height="20" alt="book" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="github" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source](https://github.com/ciaoidea/HolographiX) | Docs: [docs](docs/README.md) | Wiki: [docs](https://github.com/ciaoidea/HolographiX/wiki) | <img width="20" height="20" alt="medium" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/from-the-internet-to-holographic-networks-holo-is-not-a-download-its-convergence-a55733370ae8?postPublishedType=repub) | <img width="20" height="20" alt="podcast" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast [2025 Dec 20th](https://github.com/user-attachments/assets/bb100f1e-5b36-4697-b352-a76c88c6f9db) |

<img width="1280" alt="holographix cover" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

---

## Thesis (what this actually is)

HolographiX is a **framework that makes information resilient by construction**.

Optical holography is not a “transport trick”: it is a property of *representation*. Cut the plate and you still see the whole scene; you lose detail, not geometry. This repository engineers the same invariance in software: it defines a representation in which **any non-empty subset of fragments decodes to a coherent best-so-far estimate**, improving smoothly as more fragments arrive.

So the core claim is not “we can move data on UDP”. The claim is:

**Reliability is moved from the channel to the object.**  
The information artifact itself remains structurally interpretable under fragmentation, loss, reordering, duplication, delay, partial storage, partial computation.

Networks, filesystems, radios, object stores, model pipelines are just different ways of circulating or sampling pieces of the same resilient field.

---

## The theoretical backbone (in plain words)

HolographiX is built on three coupled ideas:

1) **Holography (distributed evidence):** each chunk carries global support; losing chunks loses detail, not geometry.  
2) **Golden spiral → torsion (holonomy memory):** the φ-driven MatriX interleave makes evidence composition path-dependent internally (non-commuting updates), leaving a torsion-like orientation state.  
3) **Chunk reasoning (evidence-first):** everything is expressed as **chunk algebra** (merge, heal, stack, prioritize, recover). The system evolves by accumulating evidence, not by global optimization or “training phases”.

Optionally, a **Lorenz attractor** can be used as a bounded chaos schedule to diversify which chunks / coefficients are sampled when uncertainty is high.

---

## From `http://` to `holo://`

The classical Internet is host-centric: DNS resolves a name to a host, HTTP pulls a byte stream from that place, and reliability is mostly a property of the channel.

HolographiX is content-centric: `holo://...` is a cue that opens a local attractor (a session). The network delivers evidence chunks from anywhere (cache, mesh peers, gateways), in any order, and the object reconstructs progressively as uncertainty collapses into a coherent best-so-far field.

Read the full note (architecture + content routing + kernel/privacy dynamics):
**[From Internet to HolographiX](docs/vision/from-internet-to-holo.md)**

---

## Abstract (operational contract)

Let `x` be a source signal (image/audio/bytes). The encoder produces `N` chunks:

```

C = E(x) = {c_1, ..., c_N}

```

Let `S` be any finite multiset of received chunks with at least one valid chunk
for the same `content_id` (duplicates and arbitrary order allowed). The decoder returns:

```

(x_hat, u, K) = D(S)

```

- `x_hat` is a full-support best-so-far reconstruction  
- `u` is an uncertainty map/curve  
- `K` is an (optional) **torsion/holonomy state**: an antisymmetric path-memory of evidence composition

Define the reference reconstruction `x* = D(C)`. For a distortion measure `d(.,.)`,
the system is designed so that, for increasing evidence, expected distortion decreases:

```

E[ d( D(S_k), x* ) ] is non-increasing in k

````

where `S_k` contains `k` distinct informative chunks sampled from `C` (after validity checks).

Missing evidence should manifest primarily as loss of detail (attenuated / missing high-frequency
coefficients), not as missing coordinates (holes in space/time).

This permutation-invariant, idempotent “any subset works” property is generalized holography.

---

## Not Just an Encoder/Decoder: A Resilient Information Field Framework

If you look at HolographiX as “a decoder that explodes information into a field and then transports it”, you miss the point. The “field” is not an intermediate representation: it is the *data structure*, the durable form of the information.

Transport is optional; **field operations are fundamental**. You can store a field, merge fields, heal a damaged field, stack fields to raise SNR, pack multiple objects into one field, prioritize transmission by gain, add recovery chunks. Those are not add-ons; they are the algebra that makes “resilient information” real.

---

## Field dynamics: torsion from the golden spiral + Lorenz exploration

### 1) Golden spiral → torsion memory (φ)

HolographiX distributes residual evidence across chunks with a **golden-ratio / golden-angle interleave** (the MatriX).
This is a deliberate φ-driven phase advance (a discrete logarithmic spiral in coefficient space):

- It avoids resonance and clumping: evidence coverage becomes quasi-uniform across support.
- It makes internal updates **non-commuting**: order can imprint a stable orientation.

We track that imprint as a torsion-like state `K`:

- same chunks, different order ⇒ same coherent reconstruction trend, but possibly different internal phase/orientation
- `K` can guide **healing**, **stacking**, **gain scheduling**, and **alignment** between partially overlapping fields

This is “torsion” in the operational sense: **history written geometrically (holonomy)**.

### 2) Lorenz attractor (optional chaos schedule)

Under heavy loss, deterministic heuristics can keep sampling redundant evidence.
HolographiX optionally uses a **bounded chaotic driver** (Lorenz attractor) to diversify which coefficients/chunks are emphasized when uncertainty is high:

- low uncertainty ⇒ stable, contractive chunk accumulation
- high uncertainty ⇒ controlled chaos that perturbs selection/healing directions, still reproducible and content-seeded

Lorenz is not the “meaning engine”; it is the **exploration budget** of the field.

### 3) Why this matches holography + holonomic brain theory

- **Holography:** distributed evidence; missing chunks = missing detail, not missing coordinates.
- **Holonomy (Pribram):** phase-sensitive distributed memory; reconstruction from partial cues.
- **HolographiX:** golden phase advance (φ) + torsion memory (`K`) + chunk algebra yields a practical holonomic representation.

---

## Mechanism (how invariance is enforced)

HolographiX uses a coarse + residual decomposition.

- The **coarse part** provides a scaffold that makes single-chunk decoding meaningful.
- The **residual** is distributed across chunks using a deterministic golden-ratio interleave (the MatriX) so that each chunk touches the whole support of the signal.

As a result, “how many chunks you have” matters far more than “which chunk IDs you have”.

HolographiX separates representation from transport. The codec/math defines how evidence is spread across chunks; the same chunks can live on UDP meshes, filesystems, object stores, delay-tolerant links, or flow directly into inference pipelines. The primitive is a stateless best-so-far field.

AI fit: best-so-far fields enable anytime inference — models can run on partial reconstructions (or on field tokens) and improve continuously as more evidence arrives. Operational loop: receive chunks -> decode best-so-far -> run model -> repeat.

---

## v3.0 (olonomic) in one sentence

v3 moves residuals into local wave bases (DCT for images, STFT for audio). Missing chunks become missing coefficients (“missing waves”), shrinking chunks while keeping graceful degradation.

---

## What’s new in 3.0 (olonomic v3)

Images: residual -> block DCT (default 8×8), JPEG-style quantization, zigzag, golden split across chunks. Missing chunks = missing waves, not missing pixels.

Audio: residual -> STFT (Hann window, overlap-add), per-bin quantization, golden split across chunks. Missing chunks = softer/detail loss, not gaps/clicks.

Metadata containers: v3 coarse payloads carry codec params (block size, quality, padding for images; n_fft/hop/quality for audio) without changing header size.

CLI flag: `--olonomic` to encode with v3. Decoding auto-detects version per chunk dir.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src          # install holo and deps (numpy, pillow)
# or run in-place: PYTHONPATH=src python3 -m holo ...
````

---

## Quick start

Note: `.holo` output directories (like `src/flower.jpg.holo/`) are generated locally and are not committed to the repository. Run the encode step before any example that reads `src/flower.jpg.holo`.

Encode / decode an image:

```bash
python3 -m holo src/flower.jpg 32
python3 -m holo src/flower.jpg.holo --output out.png
```

Encode / decode olonomic (v3):

```bash
python3 -m holo --olonomic src/flower.jpg --blocks 16 --quality 40
python3 -m holo src/flower.jpg.holo --output out.png
```

Chunk sizing: use either `TARGET_KB` (positional) or `--blocks` (explicit count); if both are given, `--blocks` wins.

Audio:

```bash
python3 -m holo /path/to/track.wav 16
python3 -m holo --olonomic /path/to/track.wav 16
python3 -m holo /path/to/track.wav.holo --output track_recon.wav
```

Try packet-sized chunks (mesh/UDP):

```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16
```

<p align="center">
  <img width="1280" alt="graded reconstruction" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" /><br/>
  <em>Graded reconstruction: fewer chunks soften detail without holes.</em>
</p>

---

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

---

## CLI cheat-sheet (core)

* `python3 -m holo INPUT [TARGET_KB]` – encode a file to `INPUT.holo`
* `python3 -m holo INPUT.holo [--max-chunks K]` – decode using up to K chunks (auto-detects v2/v3)
* `--olonomic` – use version 3 (DCT/STFT residuals); pair with `--quality Q`
* `--blocks N` – set chunk count (default keeps coarse duplicated per chunk)
* `--packet-bytes B` – set MTU budget (default 0 = no limit; increases chunk count when set)
* `--coarse-side S` (images) / `--coarse-frames F` (audio) – coarse resolution
* `--coarse-model downsample|latent_lowfreq|ae_latent` – coarse base model for v3
* `--recovery rlnc --overhead 0.25` – add recovery chunks (systematic RLNC)
* `--use-recovery` – use recovery chunks when decoding (auto if present)
* `--prefer-gain` – when decoding with `--max-chunks`, choose best-K by score
* `--write-uncertainty` – emit confidence map/curve next to the output
* `--heal` – heal a `.holo` dir (chunk-wise repair + redistribution)
* `--heal-out DIR` – output dir for healing (default: derived)
* `--heal-target-kb N` – chunk size for healing output
* `--stack dir1 dir2 ...` – stack multiple `.holo` dirs (image or audio); see `docs/kernel/stacking.md`
* `--stack-no-gauge` – disable gauge alignment for v3 stacking (pure average)
* `tnc-tx --chunk-dir DIR [--uri holo://id] --out tx.wav` – encode chunks to AFSK WAV (URI optional)
* `tnc-rx --input rx.wav --out DIR [--uri holo://id]` – decode AFSK WAV into chunks
* `tnc-wav-fix --input in.wav --out fixed.wav` – re-encode to PCM16 mono

---

## Framework CLI map

```
      Input (image / audio / arbitrary file)
                    |
                    v
         +--------------------------+
         |  Holo Codec CLI          |
         |  python3 -m holo         |
         |  encode/decode/heal      |
         +--------------------------+
                    |
              .holo chunk dir
                    |
      +-------------+------------------+
      |                                |
      v                                v
+----------------------+        +----------------------+
| Holo Net CLI         |        | Holo TNC CLI          |
| python3 -m holo net  |        | python3 -m holo tnc-* |
| UDP framing + mesh   |        | AFSK WAV modem        |
+----------------------+        +----------------------+
      |                                |
      v                                v
 UDP sockets                      Audio / Radio link
```

Notes:

* `holo://...` URIs map to `content_id = blake2s(holo://...)`.
* Net layer handles datagrams and the control plane (inventory/want).
* TNC layer turns datagrams into audio (and back).

---

## CLI help and navigation

```bash
python3 -m holo --help
python3 -m holo net --help
python3 -m holo net <command> --help
python3 -m holo tnc-tx --help
python3 -m holo tnc-rx --help
python3 -m holo tnc-wav-fix --help
```

---

## Python API highlights

```python
import holo
from holo.codec import (
    encode_image_holo_dir, decode_image_holo_dir,
    encode_audio_holo_dir, decode_audio_holo_dir,
    encode_image_olonomic_holo_dir, decode_image_olonomic_holo_dir,
    encode_audio_olonomic_holo_dir, decode_audio_olonomic_holo_dir,
    stack_image_holo_dirs,
    stack_audio_holo_dirs,
)

encode_image_olonomic_holo_dir("frame.png", "frame.holo", block_count=16, quality=60)
decode_image_holo_dir("frame.holo", "frame_recon.png")   # auto-dispatch by version

encode_audio_olonomic_holo_dir("track.wav", "track.holo", block_count=12, n_fft=256)
decode_audio_holo_dir("track.holo", "track_recon.wav")
```

---

## Advanced modes (field algebra in practice)

Recovery (RLNC): optional recovery chunks (`recovery_*.holo`) can reconstruct missing residual slices under heavy loss. Encode with `--recovery rlnc --overhead 0.25` and decode with `--use-recovery`.

Coarse models: v3 coarse is pluggable (`--coarse-model downsample|latent_lowfreq|ae_latent`). `latent_lowfreq` keeps only low-frequency DCT/STFT coefficients; `ae_latent` loads optional tiny weights from `.npz` if present.

Uncertainty output: decode with `--write-uncertainty` to produce `*_confidence.png` (images) or `*_confidence.npy` (audio) where 1.0 means fully observed.

Chunk priority: encoders write per-chunk scores and `manifest.json` ordering. Use `--prefer-gain` for best-K decode, and mesh sender priority flags to transmit high-gain chunks first.

Healing: `--heal` performs a **chunk-wise repair + redistribution** pass so degraded fields regain a clean evidence layout without changing the core “any subset works” contract.

TNC (experimental): `holo.tnc` provides a minimal AFSK modem + framing to carry `holo.net.transport` datagrams over audio.

HoloTV (experimental): `holo.tv` schedules multi-frame windows and demuxes datagrams into per-frame fields above `holo.net` and `holo.tnc`.

---

## Update summary (latest)

Recovery: systematic RLNC (`recovery_*.holo`) + GF(256) solver, optional in v3 image/audio encode/decode; mesh can send recovery chunks.

Coarse models: downsample/latent_lowfreq/ae_latent interface wired into v3 metadata, with an optional training script for AE weights.

Uncertainty: confidence maps/curves from decoder masks, new meta decode helpers, plus honest healing to attenuate uncertain regions.

Chunk priority: score-aware manifest + prefer-gain decode and mesh sender ordering support.

Healing: chunk-wise repair + redistribution with drift guards for lossy v3.

<details>
  <summary><b>Implementation status (plan checklist)</b></summary>

* [x] Review codec/field/mesh flow and implement RLNC recovery chunk format + GF(256) helpers.
* [x] Add coarse-model abstraction (downsample, latent_lowfreq, ae_latent) and store model name in v3 metadata.
* [x] Implement uncertainty tracking + meta decode helpers; integrate priority selection and manifest generation.
* [x] Add tests for recovery/uncertainty/prefer-gain/healing; update README/examples/CLI.
* [x] Run test suite and address issues.

</details>

---

## CLI guide (new features)

Recovery encode + decode:

```bash
PYTHONPATH=src python3 -m holo --olonomic src/flower.jpg --blocks 12 --recovery rlnc --overhead 0.5
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --use-recovery
```

Prefer-gain decode (best-K chunks):

```bash
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --max-chunks 4 --prefer-gain
```

Uncertainty output:

```bash
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --write-uncertainty
```

Healing (chunk-wise):

```bash
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --heal
```

Mesh sender priority + recovery:

```bash
PYTHONPATH=src python3 src/examples/holo_mesh_sender.py --uri holo://demo/flower --chunk-dir src/flower.jpg.holo --peer 127.0.0.1:5000 --priority gain --send-recovery
```

---

## TNC quickstart (experimental)

AFSK loopback example (no soundcard required):

```python
import numpy as np
from holo.tnc import AFSKModem
from holo.tnc.channel import awgn

modem = AFSKModem()
payload = b"hello field"
samples = modem.encode(payload)
samples = awgn(samples, 30.0)  # optional noise
decoded = modem.decode(samples)
assert decoded == [payload]
```

---

## Ham radio transport (HF/VHF/UHF/SHF)

HolographiX does not turn images into audio content. The audio you transmit is only a modem carrier for bytes.
WAV size is dominated by AFSK bitrate (~payload_bytes * 16 * fs / baud). To shrink WAVs, raise `--baud` or lower `--fs`,
and optionally reduce payload size with v3 (`--olonomic`, lower `--quality` / `--overhead`).

Pipeline overview:

```
image/audio -> holo.codec -> chunks (.holo)
  -> holo.net.transport (datagrams)
  -> holo.tnc (AFSK/PSK/FSK/OFDM modem)
  -> radio audio

radio audio
  -> holo.tnc -> datagrams -> chunks -> holo.codec decode
```

Why datagrams on radio:

* CRC lets you drop corrupted frames instead of smearing errors across the image.
* Loss/reordering is expected on HF; field chunks degrade in detail, not in geometry.
* Interleaving and RLNC recovery stay possible without retransmissions.

In practice you can replace the AFSK demo with any modem that yields bytes. The Holo layers above it stay unchanged.

### One-line commands (encode + TX, RX + decode)

```bash
# WAV size scales with bitrate (~payload_bytes * 16 * fs / baud); shrink by raising --baud or lowering --fs.
# Noisy band (HF-like, v3): encode -> tnc-tx
PYTHONPATH=src python3 -m holo --olonomic src/flower.jpg --blocks 12 --quality 30 --recovery rlnc --overhead 0.25

PYTHONPATH=src python3 -m holo tnc-tx --chunk-dir src/flower.jpg.holo --uri holo://noise/demo --out tx_noise.wav \
  --max-payload 320 --gap-ms 40 --preamble-len 16 --fs 9600 --baud 1200 --prefer-gain --include-recovery

# Noisy band (HF-like): tnc-rx -> decode
PYTHONPATH=src python3 -m holo tnc-rx --input tx_noise.wav --uri holo://noise/demo --out rx_noise.holo --baud 1200 --preamble-len 16

PYTHONPATH=src python3 -m holo rx_noise.holo --output rx.png --use-recovery --prefer-gain

# Clean link (VHF/UHF/SHF, v3): encode -> tnc-tx
PYTHONPATH=src python3 -m holo --olonomic src/flower.jpg --blocks 12 --quality 30 \
  && PYTHONPATH=src python3 -m holo tnc-tx --chunk-dir src/flower.jpg.holo --uri holo://clean/demo --out tx_clean.wav \
  --max-payload 512 --gap-ms 15 --preamble-len 16 --prefer-gain --fs 9600 --baud 1200

# Clean link (VHF/UHF/SHF): tnc-rx -> decode
PYTHONPATH=src python3 -m holo tnc-rx --input rx_clean.wav --uri holo://clean/demo --out rx_clean.holo --baud 1200 --preamble-len 16 \
  && PYTHONPATH=src python3 -m holo rx_clean.holo --output rx.png --prefer-gain
```

Loopback tip (no radio): use `tx_noise.wav` as `rx_noise.wav` to test the full chain.

---

## HoloTV quickstart (experimental)

Schedule chunks across a window of frames and feed them to a receiver:

```python
from pathlib import Path

from holo.tv import HoloTVWindow, HoloTVReceiver
from holo.cortex.store import CortexStore

window = HoloTVWindow.from_chunk_dirs(
    "holo://tv/demo",
    ["frames/f000.holo", "frames/f001.holo"],
    prefer_gain=True,
)

store = CortexStore("tv_store")
rx = HoloTVReceiver("holo://tv/demo", store, frame_indices=[0, 1])

for datagram in window.iter_datagrams():
    rx.push_datagram(datagram)

frame0_dir = rx.chunk_dir_for_frame(0)
print("frame 0 chunks:", sorted(Path(frame0_dir).glob("chunk_*.holo")))
```

---

## Olonomic v3 details (operational)

Images: residual = img − coarse_up -> pad to block size -> DCT-II (ortho) per block/channel -> JPEG-style quant (quality 1-100) -> zigzag -> int16 -> golden permutation -> zlib per chunk. Missing chunks zero coefficients; recon via dequant + IDCT + coarse.

Audio: residual = audio − coarse_up -> STFT (sqrt-Hann, hop=n_fft/2 default) -> scale by n_fft -> per-bin quant steps grow with freq (quality 1-100) -> int16 (Re/Im interleaved) -> golden permutation -> zlib per chunk. Recon via dequant, ISTFT overlap-add, coarse + residual.

Metadata: packed inside coarse payload (PNG + OLOI_META for images; zlib coarse + OLOA_META for audio) so headers stay backward-compatible.

Field operations are first-class: decode partially at any time, heal to restore a clean distribution, stack exposures to raise SNR, and pack/extract multiple objects into one field.

---

## Cite this work (preferred)

If you use HolographiX ideas (information fields, golden-spiral interleave, torsion/holonomy memory, Lorenz exploration, chunk healing, stacking), please cite:

* Rizzo, A. (2025). *HolographiX: Holographic Information MatriX for Resilient Content Diffusion in Networks* (V3.0). Zenodo. DOI: 10.5281/zenodo.18017872
* Rizzo, A. (2025). *HolographiX: From Fragile Streams to Information Fields*. ISBN-13: 979-8278598534

---

## Acknowledgements (healing)

Thanks to **Stefan Hamann** for a practical suggestion on the *healing* workflow (treat healing as an iterative self-consistency pass). In HolographiX this is implemented strictly as **chunk-wise repair + redistribution**, consistent with the “any subset works” contract.

---

## References

* Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. [https://doi.org/10.1016/0001-6918(86)90062-4](https://doi.org/10.1016/0001-6918%2886%2990062-4)
* Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4
* Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).
* Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9
* Rizzo, A. (2023). *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. DOI: 10.4236/apm.2023.139038
* Rizzo, A. (2025). *HolographiX: Holographic Information MatriX for Resilient Content Diffusion in Networks* (V1.7). Zenodo. DOI: 10.5281/zenodo.18017872
* Rizzo, A. (2025). *HolographiX: From Fragile Streams to Information Fields*. ISBN-13: 979-8278598534

---

## License

HolographiX is licensed under the GNU Affero General Public License v3.0 (AGPLv3). See `LICENSE`. Copyright (C) 2025 Alessandro Rizzo.

<p align="center">
  © 2025 <a href="https://holographix.io">holographix.io</a>
</p>

