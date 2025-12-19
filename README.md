# <img width="36" height="36" alt="HolographiX logo" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX
## V3.0 — Information fields for lossy, unordered worlds

| <img width="20" height="20" alt="paper" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464) | <img width="20" height="20" alt="book" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="github" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source](https://github.com/ciaoidea/HolographiX) | <img width="20" height="20" alt="medium" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/the-best-so-far-economy-why-i-m-betting-on-fields-not-streams-093b176be1e8) | <img width="20" height="20" alt="podcast" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast [2025 Dec 17th](https://github.com/user-attachments/assets/a3b973a8-d046-4bea-8516-bd8494601437) |

<img width="1280" alt="holographix cover" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

HolographiX is a field-first representation layer: it turns data into interchangeable chunks such that any non‑empty subset decodes to a coherent best‑so‑far estimate that refines smoothly as more chunks arrive. This is not just media compression — it’s a way to move, store, fuse, and “heal” information in hostile conditions (loss, reordering, partial availability). Think fields, not streams: detail fades when chunks are missing, never by punching holes in space or time.

HolographiX separates representation from transport. The codec/math defines how evidence is spread across chunks; the same chunks can live on UDP meshes, filesystems, object stores, delay-tolerant links, or flow directly into inference pipelines. Networks are one use-case; the core primitive is a stateless best‑so‑far field.

AI fit: best‑so‑far fields enable anytime inference — models can run on partial reconstructions (or on field tokens) and improve continuously as more evidence arrives. Operational loop: **receive chunks → decode best‑so‑far → run model → repeat** (e.g., call a VLM with `max_chunks=k`, then re‑decode with more chunks and re‑query).

- **Coarse + residual**: a tiny coarse thumbnail/envelope plus a distributed residual.
- **MatriX (golden interleave)**: deterministic golden-ratio permutation so every chunk touches the whole signal.
- **Stateless, deterministic**: no session/state needed; chunks are interchangeable.
- **New in 3.0 (olonomic v3)**: residuals live in local wave bases (DCT for images, STFT for audio), drastically shrinking chunk sizes while keeping graceful degradation.

## What’s new in 3.0 (olonomic v3)
- **Images**: residual → block DCT (default 8×8), JPEG‑style quantization, zigzag, golden split across chunks. Missing chunks = missing waves, not missing pixels.
- **Audio**: residual → STFT (Hann window, overlap‑add), per‑bin quantization, golden split across chunks. Missing chunks = softer/detail loss, not gaps/clicks.
- **Metadata containers**: v3 coarse payloads carry codec params (block size, quality, padding for images; n_fft/hop/quality for audio) without changing header size.
- **CLI flag**: `--olonomic` to encode with v3. Decoding auto‑detects version per chunk dir.

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src          # install holo and deps (numpy, pillow)
# or run in-place: PYTHONPATH=src python3 -m holo ...
```

## Quick start
Note: `.holo` output directories (like `src/flower.jpg.holo/`) are generated locally and are not committed to the repo. Run the encode step before any example that reads `src/flower.jpg.holo`.
Encode / decode an image:
```bash
python3 -m holo src/flower.jpg 32                 # v2 (pixel residual), ~32 KB chunks
python3 -m holo src/flower.jpg.holo --output out.png
```
Encode / decode olonomic (v3):
```bash
python3 -m holo --olonomic src/flower.jpg --blocks 16 --quality 40  # DCT residual, wave-based loss (smaller than v2)
python3 -m holo src/flower.jpg.holo --output out.png
```
Chunk sizing: use either `TARGET_KB` (positional) or `--blocks` (explicit count); if both are given, `--blocks` wins.
Audio:
```bash
python3 -m holo /path/to/track.wav 16             # v2
python3 -m holo --olonomic /path/to/track.wav 16  # v3 STFT
python3 -m holo /path/to/track.wav.holo --output track_recon.wav
```
Try packet‑sized chunks (mesh/UDP):
```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16   # enables tiny chunks; expect many files
```

<p align="center">
  <img width="1280" alt="graded reconstruction" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" /><br/>
  <em>Graded reconstruction: fewer chunks soften detail without holes.</em>
</p>

## CLI cheat‑sheet
- `python3 -m holo INPUT [TARGET_KB]` – encode file to `INPUT.holo`
- `python3 -m holo INPUT.holo [--max-chunks K]` – decode using up to K chunks (auto-detects v2/v3)
- `--olonomic` – use version 3 (DCT/STFT residuals); pair with `--quality Q`
- `--blocks N` – set chunk count (default keeps coarse duplicated per chunk)
- `--packet-bytes B` – set MTU budget (default 0 = no limit; increases chunk count when set)
- `--coarse-side S` (images) / `--coarse-frames F` (audio) – coarse resolution
- `--coarse-model downsample|latent_lowfreq|ae_latent` – coarse base model for v3
- `--recovery rlnc --overhead 0.25` – add recovery chunks (systematic RLNC)
- `--use-recovery` – use recovery chunks when decoding (auto if present)
- `--prefer-gain` – when decoding with `--max-chunks`, choose best-K by score
- `--write-uncertainty` – emit confidence map/curve next to the output
- `--heal` / `--heal-fixed-point` – heal a .holo dir (one-step or fixed-point)
- `--heal-out DIR` – output dir for healing (default: derived)
- `--heal-target-kb N` – chunk size for healing output
- `--stack dir1 dir2 ...` – stack multiple image .holo dirs (average recon)
- `tnc-tx --chunk-dir DIR [--uri holo://id] --out tx.wav` – encode chunks to AFSK WAV (URI optional)
- `tnc-rx --input rx.wav --out DIR [--uri holo://id]` – decode AFSK WAV into chunks
- `tnc-tx ... --fs 9600 --baud 1200 --max-chunks 4` – reduce WAV size for quick tests (size scales with `fs/baud`)
- `tnc-wav-fix --input in.wav --out fixed.wav` – re-encode to PCM16 mono

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

## Advanced modes
- **Recovery (RLNC)**: optional recovery chunks (`recovery_*.holo`) can reconstruct missing residual slices under heavy loss. Encode with `--recovery rlnc --overhead 0.25` and decode with `--use-recovery`.
- **Coarse models**: v3 coarse is now pluggable (`--coarse-model downsample|latent_lowfreq|ae_latent`). `latent_lowfreq` keeps only low‑frequency DCT/STFT coefficients; `ae_latent` loads optional tiny weights from `.npz` if present.
- **Uncertainty output**: decode with `--write-uncertainty` to produce `*_confidence.png` (images) or `*_confidence.npy` (audio) where 1.0 = fully observed.
- **Chunk priority**: encoders write per‑chunk scores and `manifest.json` ordering. Use `--prefer-gain` for best‑K decode, and mesh sender priority flags to transmit high‑gain chunks first.
- **Fixed‑point healing**: `Field.heal_fixed_point(...)` iterates healing until deltas stabilize, with drift guards for lossy v3.
- **CLI healing**: use `--heal` or `--heal-fixed-point` on a `.holo` directory to re-encode the current best‑so‑far.
- **TNC (experimental)**: `holo.tnc` provides a minimal AFSK modem + framing to carry `holo.net.transport` datagrams over audio.
- **HoloTV (experimental)**: `holo.tv` schedules multi-frame windows and demuxes datagrams into per-frame fields above `holo.net` and `holo.tnc`.

## Update summary (latest)
- **Recovery**: systematic RLNC (`recovery_*.holo`) + GF(256) solver, optional in v3 image/audio encode/decode; mesh can send recovery chunks.
- **Coarse models**: downsample/latent_lowfreq/ae_latent interface wired into v3 metadata, with optional training script for AE weights.
- **Uncertainty**: confidence maps/curves from decoder masks, new meta decode helpers, and honest healing to attenuate uncertain regions.
- **Chunk priority**: score-aware manifest + prefer-gain decode and mesh sender ordering.
- **Healing**: fixed-point healing loop with convergence metric and drift guards.

## Updated plan (implementation status)
- [x] Review codec/field/mesh flow and implement RLNC recovery chunk format + GF(256) helpers.
- [x] Add coarse-model abstraction (downsample, latent_lowfreq, ae_latent) and store model name in v3 metadata.
- [x] Implement uncertainty tracking + meta decode helpers; integrate priority selection and manifest generation.
- [x] Add tests for recovery/uncertainty/prefer-gain/fixed-point healing; update README/examples/CLI.
- [x] Run test suite and address issues.

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
Healing (one-step and fixed-point):
```bash
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --heal
PYTHONPATH=src python3 -m holo src/flower.jpg.holo --heal-fixed-point --heal-iters 4 --heal-tol 1e-3
```
Mesh sender priority + recovery:
```bash
PYTHONPATH=src python3 src/examples/holo_mesh_sender.py --uri holo://demo/flower --chunk-dir src/flower.jpg.holo --peer 127.0.0.1:5000 --priority gain --send-recovery
```

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

## Ham radio transport (HF/VHF/UHF/SHF)
Holo does not turn images into audio content. The audio you transmit is only a modem carrier for bytes.
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
- CRC lets you drop corrupted frames instead of smearing errors across the image.
- Loss/reordering is expected on HF; field chunks degrade in detail, not in geometry.
- Interleaving and RLNC recovery stay possible without retransmissions.

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

### On-air workflow (TX -> RX)
1) Encode the image/audio into `.holo` chunks.
2) Run `tnc-tx` to build a WAV (baseband audio).
3) Feed WAV audio into the radio (line-in/IF preferred).
4) Record RX audio into a WAV.
5) Run `tnc-rx` to rebuild chunks, then decode the image/audio.

Suggested parameter table (AFSK, conservative defaults):

| Link quality | --baud | --fs | --max-payload | --gap-ms | --include-recovery | Compression/AGC |
| --- | --- | --- | --- | --- | --- | --- |
| Noisy/variable (HF) | 1200 | 9600 | 320 | 40 | yes | OFF |
| Clean link (VHF/UHF/SHF) | 1200 | 9600 | 512 | 15 | optional | OFF |

Notes:
- Use line-in/IF audio from the rig or SDR when possible; avoid acoustic coupling.
- Disable AGC/compression when you can; keep levels below clipping.
- `--max-payload` and `--gap-ms` trade throughput vs robustness; tune for your link budget.
- If you run multiple streams, pass an explicit `--uri` on TX and RX to avoid mixing.
- Size rule of thumb: `wav_bytes ~ payload_bytes * (16 * fs / baud)`; `--gap-ms` and `--preamble-len` add overhead. Lower `fs`, raise `baud`, or limit `--max-chunks` for smaller files.
- Example: 600 KB payload at `fs=9600`, `baud=1200` gives ~600 * 128 = 76 MB before gaps/preamble; overhead can push this well past 100 MB.
- `tnc-rx` defaults to best-effort PCM16 decode; disable with `--no-force-pcm16` if needed.
- If you edited or trimmed a WAV and the header breaks, fix it with:
  `PYTHONPATH=src python3 -m holo tnc-wav-fix --input rx.wav --out rx_pcm.wav`
- If the file is badly corrupted, force raw decode:
  `PYTHONPATH=src python3 -m holo tnc-rx --input rx.wav --raw-s16le --raw-fs 48000 --out rx.holo`
 

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

## Olonomic v3 details (operational)
- **Images**: residual = img − coarse_up → pad to block size → DCT‑II (ortho) per block/channel → JPEG‑style quant (quality 1‑100) → zigzag → int16 → golden permutation → zlib per chunk. Missing chunks zero coefficients; recon via dequant + IDCT + coarse.
- **Audio**: residual = audio − coarse_up → STFT (sqrt-Hann, hop=n_fft/2 default) → scale by n_fft → per‑bin quant steps grow with freq (quality 1‑100) → int16 (Re/Im interleaved) → golden permutation → zlib per chunk. Recon via dequant, ISTFT overlap‑add, coarse + residual.
- **Metadata**: packed inside coarse payload (PNG + OLOI_META for images; zlib coarse + OLOA_META for audio) so headers stay backward‑compatible.

Field operations are first-class: decode partially at any time, heal to restore a clean distribution, stack exposures to raise SNR, and pack/extract multiple objects into one field.

## Repository map
```
README.md                  top-level overview (this file)
src/pyproject.toml         packaging for editable install
src/requirements.txt       runtime deps (numpy, pillow)

src/holo/                  core library
  codec.py                 codecs v1/v2/v3 (image/audio), chunk scoring, recovery hooks
  recovery.py              GF(256) RLNC recovery chunks + solver
  __main__.py              CLI entry (codec + tnc-tx/tnc-rx)
  __init__.py              public API surface
  container.py             multi-object packing/unpacking
  field.py                 field tracking + healing (fixed-point)
  cortex/                  storage helpers (store.py backend)
  net/                     transport + mesh + content IDs
    transport.py           datagram framing/reassembly (HODT/HOCT)
    mesh.py                UDP mesh sender/receiver + priority order
    arch.py                content_id helpers
  models/                  coarse model abstraction (downsample/latent_lowfreq/ae_latent)
  tnc/                     modem + framing + WAV CLI
    afsk.py                AFSK modem
    frame.py               framing + CRC
    cli.py                 tnc-tx/tnc-rx WAV helpers
  tv/                      HoloTV scheduling + demux helpers
  mind/                    stubs/placeholders for higher-layer logic

src/examples/              runnable demos (encode/decode, mesh_loopback, heal, pack/extract, benchmarks)
src/tests/                 unit tests (round-trip, recovery, tnc, tv, healing)
src/codec_simulation/      React/Vite control deck for codec exploration (optional)
src/docs/                  Global Holographic Network guide (mesh/INV-WANT, DTN, examples for sensor fusion/AI/maps)
src/infra/                 containerlab lab + netem/benchmark configs
src/systemd/               sample systemd units for mesh sender/receiver/node
src/tools/                 offline tools (e.g., AE coarse training/export)
```

## Layering map (at a glance)
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

## Testing
```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -p 'test_*.py'
```
- Includes round‑trip, PSNR/MSE monotonicity for olonomic v3, and size guard regression.

Network/mesh smoke test (loopback UDP using `holo://` content IDs):
```bash
PYTHONPATH=src python3 src/examples/mesh_loopback.py
# emits galaxy.jpg chunks on 127.0.0.1, stores in src/examples/out/store_b/...,
# and writes a reconstructed image to src/examples/out/galaxy_mesh_recon.png
```

Other functional checks (examples):
```bash
PYTHONPATH=src python3 src/examples/heal_demo.py         # damages chunks, heals via Field.heal_to, writes healed recon
PYTHONPATH=src python3 src/examples/pack_and_extract.py  # packs multiple objects into one field, then extracts after chunk loss
```

## Results snapshot
- `src/galaxy.jpg`, coarse-side=16, v3 command: `python3 -m holo --olonomic src/galaxy.jpg --blocks 16 --quality 40 --packet-bytes 0` → total ~0.35 MB (coherent single-chunk recon). Same settings v2 pixel residuals: ~1.69 MB. Visual quality comparable; v3 degrades as “missing waves”, not holes.

<p align="center">
  <img width="1280"  alt="photon collector" src="https://github.com/user-attachments/assets/c2b939d1-8911-4381-8bd7-a93e29f5401c" /><br/>
  <em>Photon-collector stacking: multiple exposures reinforce structure over noise.</em>
</p>

## Design principles
- **Interchangeability by construction**: golden permutation ensures quality depends mostly on chunk count, not chunk IDs.
- **Graceful loss**: missing chunks zero high‑freq waves instead of creating spatial/temporal holes.
- **Stateless decode**: any subset of valid chunks decodes without coordination.
- **Transport‑agnostic**: codec math is separate from mesh/UDP; use your own transport if needed.

<p align="center">
  <img width="1280"  alt="psnr curves" src="https://github.com/user-attachments/assets/e8c700f2-e5b6-424b-a848-a230294e8269" /><br/>
  <em>PSNR vs received chunks: quality rises smoothly; variance stays low.</em>
</p>

## References

- Pribram, K. H. & Carlton, E. H. (1986). *Holonomic brain theory in imaging and object perception*. Acta Psychologica, 63(2), 175–210. [https://doi.org/10.1016/0001-6918(86)90062-4](https://doi.org/10.1016/0001-6918%2886%2990062-4)
- Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing*. Hillsdale, NJ: Lawrence Erlbaum Associates. ISBN 978-0-89859-995-4
- Bohm, D. (1980). *Wholeness and the Implicate Order*. London: Routledge (Routledge Classics ed. 2002, ISBN 978-0-415-28979-5).
- Bohm, D. & Hiley, B. J. (1993). *The Undivided Universe: An Ontological Interpretation of Quantum Theory*. London: Routledge. ISBN 978-0-415-12185-9
- Rizzo, A. *The Golden Ratio Theorem*, Applied Mathematics, 14(09), 2023. DOI: [10.4236/apm.2023.139038](https://doi.org/10.4236/apm.2023.139038)
- Rizzo, A. (2025). *HolographiX: Holographic Information MatriX for Resilient Content Diffusion in Networks* (v1.6.2). DOI: [10.5281/zenodo.17957464](https://zenodo.org/records/17957464)
- Rizzo, A. (2025). *HolographiX: From Fragile Streams to Information Fields*. [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD)

<p align="center">
  © 2025 <a href="https://holographix.io">holographix.io</a>
</p>
