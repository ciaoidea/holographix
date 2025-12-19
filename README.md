# <img width="36" height="36" alt="HolographiX logo" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX
## V3.0 — Information fields for lossy, unordered worlds

| <img width="20" height="20" alt="paper" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464) | <img width="20" height="20" alt="book" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="github" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source](https://github.com/ciaoidea/HolographiX) | <img width="20" height="20" alt="medium" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/the-best-so-far-economy-why-i-m-betting-on-fields-not-streams-093b176be1e8) | <img width="20" height="20" alt="podcast" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast [2025 Dec 17th](https://github.com/user-attachments/assets/a3b973a8-d046-4bea-8516-bd8494601437) |

<img width="1280" alt="holographix cover" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

HolographiX is a field-first representation layer: it turns data into interchangeable chunks such that any non‑empty subset decodes to a coherent best‑so‑far estimate that refines smoothly as more chunks arrive. This is not just media compression — it’s a way to move, store, fuse, and “heal” information in hostile conditions (loss, reordering, partial availability). Think fields, not streams: detail fades when chunks are missing, never by punching holes in space or time.

HolographiX separates representation from transport. The codec/math defines how evidence is spread across chunks; the same chunks can live on UDP meshes, filesystems, object stores, delay-tolerant links, or flow directly into inference pipelines. Networks are one use-case; the core primitive is a stateless best‑so‑far field.

AI fit: best‑so‑far fields enable anytime inference — models can run on partial reconstructions (or on field tokens) and improve continuously as more evidence arrives.

- **Coarse + residual**: a tiny coarse thumbnail/envelope plus a distributed residual.
- **Golden interleaving**: residual samples are permuted so each chunk touches the whole signal.
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
Encode / decode an image:
```bash
python3 -m holo src/flower.jpg 32                 # v2 (pixel residual), ~32 KB chunks
python3 -m holo src/flower.jpg.holo --output out.png
```
Encode / decode olonomic (v3):
```bash
python3 -m holo --olonomic src/flower.jpg 16      # DCT residual, wave-based loss
python3 -m holo src/flower.jpg.holo --output out.png
```
Audio:
```bash
python3 -m holo /path/to/track.wav 16             # v2
python3 -m holo --olonomic /path/to/track.wav 16  # v3 STFT
python3 -m holo /path/to/track.wav.holo --output track_recon.wav
```
Try packet‑sized chunks (mesh/UDP):
```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16
```

## CLI cheat‑sheet
- `python3 -m holo INPUT [TARGET_KB]` – encode file to `INPUT.holo`
- `python3 -m holo INPUT.holo [--max-chunks K]` – decode using up to K chunks
- `--olonomic` – use version 3 (DCT/STFT residuals)
- `--blocks N` or `--packet-bytes B` – control chunk count/size
- `--coarse-side S` (images) / `--coarse-frames F` (audio) – coarse resolution
- `--stack dir1 dir2 ...` – stack multiple image .holo dirs (average recon)

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

## Olonomic v3 details (operational)
- **Images**: residual = img − coarse_up → pad to block size → DCT‑II (ortho) per block/channel → JPEG‑style quant (quality 1‑100) → zigzag → int16 → golden permutation → zlib per chunk. Missing chunks zero coefficients; recon via dequant + IDCT + coarse.
- **Audio**: residual = audio − coarse_up → STFT (sqrt-Hann, hop=n_fft/2 default) → scale by n_fft → per‑bin quant steps grow with freq (quality 1‑100) → int16 (Re/Im interleaved) → golden permutation → zlib per chunk. Recon via dequant, ISTFT overlap‑add, coarse + residual.
- **Metadata**: packed inside coarse payload (PNG + OLOI_META for images; zlib coarse + OLOA_META for audio) so headers stay backward‑compatible.

## Repository map
```
README.md                  top-level overview (this file)
src/pyproject.toml         packaging for editable install
src/requirements.txt       runtime deps (numpy, pillow)

src/holo/                  core library
  codec.py                 codecs v1/v2/v3 (image/audio), headers, golden interleave
  __main__.py              CLI entry
  __init__.py              public API surface
  container.py             multi-object packing/unpacking
  field.py                 field tracking + healing
  cortex/                  storage helpers (store.py backend)
  net/                     transport, mesh, arch (content IDs), datagram framing
  models/, mind/           stubs/placeholders for higher-layer logic

src/examples/              runnable demos (encode/decode, mesh_loopback, heal, pack/extract, benchmarks)
src/tests/                 unit tests (round-trip, PSNR/MSE monotonicity, size guards)
src/codec_simulation/      React/Vite control deck for codec exploration (optional)
src/docs/                  Global Holographic Network guide (mesh/INV-WANT, DTN, examples for sensor fusion/AI/maps)
src/infra/                 containerlab lab + netem/benchmark configs
src/systemd/               sample systemd units for mesh sender/receiver/node
```

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
- `src/galaxy.jpg` @ block_count=16: v2 total ~1.69 MB, v3 (DCT) total ~0.35 MB with coherent single‑chunk recon.

## Design principles
- **Interchangeability by construction**: golden permutation ensures quality depends mostly on chunk count, not chunk IDs.
- **Graceful loss**: missing chunks zero high‑freq waves instead of creating spatial/temporal holes.
- **Stateless decode**: any subset of valid chunks decodes without coordination.
- **Transport‑agnostic**: codec math is separate from mesh/UDP; use your own transport if needed.

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
