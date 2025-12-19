# HolographiX (v1.4) – Holographic codec for lossy, unordered worlds

HolographiX turns images and audio into holographic chunks: every non‑empty subset decodes to a coherent “best‑so‑far” view that refines smoothly as more chunks arrive. Think fields, not streams: detail fades when chunks are missing, never by punching holes in space or time.

- **Coarse + residual**: a tiny coarse thumbnail/envelope plus a distributed residual.
- **Golden interleaving**: residual samples are permuted so each chunk touches the whole signal.
- **Stateless, deterministic**: no session/state needed; chunks are interchangeable.
- **New in 1.4 (olonomic v3)**: residuals live in local wave bases (DCT for images, STFT for audio), drastically shrinking chunk sizes while keeping graceful degradation.

## What’s new (olonomic v3)
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

## Repository map (essentials)
```
src/holo/codec.py      core codecs (v1/v2/v3), headers, golden interleave
src/holo/__main__.py   CLI entry
src/holo/__init__.py   public API exports
src/tests/             unit tests (PSNR/MSE monotonicity, size guards)
src/examples/          ready-to-run demos and mesh helpers
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

## Results snapshot
- `src/galaxy.jpg` @ block_count=16: v2 total ~1.69 MB, v3 (DCT) total ~0.35 MB with coherent single‑chunk recon.

## Design principles
- **Interchangeability by construction**: golden permutation ensures quality depends mostly on chunk count, not chunk IDs.
- **Graceful loss**: missing chunks zero high‑freq waves instead of creating spatial/temporal holes.
- **Stateless decode**: any subset of valid chunks decodes without coordination.
- **Transport‑agnostic**: codec math is separate from mesh/UDP; use your own transport if needed.

## License & paper
Paper DOI: [10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464)  
Source: https://github.com/ciaoidea/HolographiX  
License: see `LICENSE`
