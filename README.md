# <img width="36" height="36" alt="HolographiX logo" src="https://github.com/user-attachments/assets/d7b26ef6-4645-4add-8ab6-717eb2fb12f2" /> HolographiX: Holographic Information MatriX
## V3.0 — Information fields for lossy, unordered worlds

| <img width="20" height="20" alt="paper" src="https://github.com/user-attachments/assets/5cb70ee6-e6f7-4c5e-95b5-95d4e306c877" /> Paper [DOI: 10.5281/zenodo.17957464](https://doi.org/10.5281/zenodo.17957464) | <img width="20" height="20" alt="book" src="https://github.com/user-attachments/assets/264bb318-20b2-4982-a4d0-f7e5373985f0" /> Book: [ISBN-13: 979-8278598534](https://www.amazon.com/dp/B0G6VQ3PWD) | <img width="20" height="20" alt="github" src="https://github.com/user-attachments/assets/e939c63a-fa18-4363-abfe-ed1e6a2f5afc" /> GitHub: [source](https://github.com/ciaoidea/HolographiX) | <img width="20" height="20" alt="medium" src="https://github.com/user-attachments/assets/7ca2ea42-1fac-4fc0-a66f-cf5a5524fe1f" /> Medium [Article](https://ciaoidea.medium.com/the-best-so-far-economy-why-i-m-betting-on-fields-not-streams-093b176be1e8) | <img width="20" height="20" alt="podcast" src="https://github.com/user-attachments/assets/986237bf-7a4f-4b14-91c4-b144cd1b48d2" /> Podcast [2025 Dec 17th](https://github.com/user-attachments/assets/a3b973a8-d046-4bea-8516-bd8494601437) |

<img width="1280" alt="holographix cover" src="https://github.com/user-attachments/assets/ae95ff1f-b15f-46f3-bf1c-bebab868b851" />

## Abstract

HolographiX is a representation layer that converts a signal (image, audio, or arbitrary bytes) into a set of interchangeable chunks. The defining constraint is operational, not aesthetic: any non-empty subset of chunks must decode to a coherent *best-so-far* reconstruction, and quality must improve smoothly as more chunks arrive, even when chunks are lost, reordered, duplicated, or delayed.

This is achieved by separating *representation* from *transport*. The codec defines how evidence is distributed across chunks; transport can be UDP meshes, filesystems, object stores, delay-tolerant links, audio/radio modems, or direct ingestion into inference pipelines. The primitive is a stateless “field” you can decode at any time.

A compact way to state the goal is:

Given an original signal x, encode into chunks C = {c_1..c_N}. For any subset S ⊆ C with S != ∅, the decoder produces x_hat(S). As |S| increases, x_hat(S) refines without creating spatial/temporal holes; missing information expresses itself mainly as loss of detail (high-frequency energy), not missing coordinates.

## What “field-first” means in practice

A stream assumes order and continuity; a field assumes partial evidence. HolographiX is built so that “how many chunks you have” matters far more than “which specific chunks you have”. That invariance is enforced by construction via a deterministic golden-ratio permutation (the MatriX / golden interleave), spreading each chunk’s contribution across the whole signal support.

The codec follows a coarse + residual split. The coarse component is duplicated (or otherwise made easy to obtain early) so a single chunk can already decode to something coherent. The residual is distributed so that missing chunks progressively suppress detail rather than carving holes.

A useful mental model for computation is an anytime loop:

receive chunks -> decode best-so-far -> run model/inference -> repeat

This works because the decoder is stateless: no session handshake, no “packet 17 must arrive before packet 18”.

## Versions: v2 and v3 (olonomic)

Versioning matters because the “shape of degradation” depends on the residual domain.

In v2, residuals are pixel/sample-domain deltas. It is simple and robust, but chunk sizes are larger at comparable perceptual quality.

In v3 (“olonomic”), residuals live in local wave bases. For images this is block DCT; for audio this is STFT. Under loss, you lose waves (coefficients), not pixels/samples. The result is smaller chunks for the same operating point and a degradation mode that tends to look like detail softening instead of geometric corruption.

The CLI flag `--olonomic` selects v3 at encode time; decoding auto-detects the version in a `.holo` directory.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./src          # install holo and deps (numpy, pillow)
# or run in-place: PYTHONPATH=src python3 -m holo ...
````

## Quick start

Note: `.holo` output directories (for example `src/flower.jpg.holo/`) are generated locally and are not committed to the repo. Run an encode step before any example that reads `*.holo`.

Image encode/decode (v2, pixel residuals):

```bash
python3 -m holo src/flower.jpg 32                 # encode to src/flower.jpg.holo/, ~32 KB chunks
python3 -m holo src/flower.jpg.holo --output out.png
```

Image encode/decode (v3, olonomic DCT residuals):

```bash
python3 -m holo --olonomic src/flower.jpg --blocks 16 --quality 40
python3 -m holo src/flower.jpg.holo --output out.png
```

Chunk sizing: you can provide `TARGET_KB` positionally or set `--blocks N`. If both are given, `--blocks` wins.

Audio (WAV):

```bash
python3 -m holo /path/to/track.wav 16
python3 -m holo --olonomic /path/to/track.wav 16
python3 -m holo /path/to/track.wav.holo --output track_recon.wav
```

Packet-sized chunks (mesh/UDP). This trades filesystem object count for MTU friendliness:

```bash
python3 -m holo src/flower.jpg 1 --packet-bytes 1136 --coarse-side 16
```

<p align="center">
  <img width="1280" alt="graded reconstruction" src="https://github.com/user-attachments/assets/b1cd73a9-e4cc-43df-b528-d5c1c184ad52" /><br/>
  <em>Graded reconstruction: fewer chunks soften detail without holes.</em>
</p>

## CLI: what you can do

The codec CLI is `python3 -m holo`. It can encode, decode, heal, and stack `.holo` directories. Networking (`holo.net`) and audio/radio modem transport (`holo.tnc`) sit above the same chunk representation.

A compact map of the layers:

```
holo.codec   -> chunk bytes (field representation)
holo.net     -> datagrams (framing + mesh)
holo.tnc     -> audio/radio modem (AFSK/FSK/PSK/etc)
holo.tv      -> multi-frame scheduling (HoloTV windows)
```

Framework overview:

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

CLI help is your friend (the option surface is intentionally explicit):

```bash
python3 -m holo --help
python3 -m holo net --help
python3 -m holo net <command> --help
python3 -m holo tnc-tx --help
python3 -m holo tnc-rx --help
python3 -m holo tnc-wav-fix --help
```

### Core flags (selected)

| Flag                                                     | Meaning                                                                             |            |                                 |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------- | ------------------------------- |
| `--olonomic`                                             | Encode using v3 residuals (DCT for images, STFT for audio).                         |            |                                 |
| `--quality Q`                                            | Quantization strength (1–100). Lower = smaller chunks / more loss.                  |            |                                 |
| `--blocks N`                                             | Target number of chunks; overrides positional `TARGET_KB` when both are present.    |            |                                 |
| `--packet-bytes B`                                       | MTU budget for datagrams; increases chunk count to fit `B`.                         |            |                                 |
| `--coarse-side S` (images) / `--coarse-frames F` (audio) | Coarse resolution.                                                                  |            |                                 |
| `--max-chunks K`                                         | Decode using up to K chunks (any subset); useful for “anytime” decoding.            |            |                                 |
| `--prefer-gain`                                          | When `--max-chunks` is used, pick the best K chunks by score instead of first K.    |            |                                 |
| `--recovery rlnc --overhead α`                           | Add systematic RLNC recovery chunks with overhead α (e.g. 0.25).                    |            |                                 |
| `--use-recovery`                                         | Use recovery chunks during decode (auto if present in many workflows).              |            |                                 |
| `--write-uncertainty`                                    | Emit confidence outputs (maps/curves) from decoder observation masks.               |            |                                 |
| `--heal` / `--heal-fixed-point`                          | Re-encode a `.holo` directory from its current best-so-far (one-step or iterative). |            |                                 |
| `--stack dir1 dir2 ...`                                  | Stack multiple `.holo` dirs (images) by averaging reconstructions.                  |            |                                 |
| `--coarse-model downsample                               | latent_lowfreq                                                                      | ae_latent` | Select coarse generator for v3. |

### Command groups

| Command                       | Description                                                               |
| ----------------------------- | ------------------------------------------------------------------------- |
| `python3 -m holo`             | Codec CLI: encode, decode, heal, stack; reads/writes `.holo` directories. |
| `python3 -m holo net ...`     | Datagram framing/reassembly utilities and UDP mesh helpers.               |
| `python3 -m holo tnc-tx`      | Encode datagrams/chunks to AFSK WAV for audio/radio links.                |
| `python3 -m holo tnc-rx`      | Decode AFSK WAV back into chunk directories.                              |
| `python3 -m holo tnc-wav-fix` | Normalize WAV/raw PCM to PCM16 mono for robust decoding.                  |

If you want the long net/mesh subcommand inventory in the README, it still exists below but is tucked away to keep the top-level readable.

<details>
  <summary><b>Transport/mesh subcommand inventory</b></summary>

| Command                                     | Description                                             |
| ------------------------------------------- | ------------------------------------------------------- |
| `python3 -m holo net constants`             | Print transport constants (MAGIC, CONTROL_MAGIC, etc.). |
| `python3 -m holo net norm-uri`              | Normalize a `holo://` URI (trim).                       |
| `python3 -m holo net id-bytes`              | `content_id` bytes from URI (hex/base64/raw).           |
| `python3 -m holo net id-hex`                | `content_id` hex from URI.                              |
| `python3 -m holo net stream-id`             | `content_id` bytes from stream id + frame index.        |
| `python3 -m holo net iter-datagrams`        | Split a chunk into transport datagrams.                 |
| `python3 -m holo net send-chunk`            | Send a chunk as UDP datagrams.                          |
| `python3 -m holo net encode-inventory`      | Build control-plane INVENTORY datagram.                 |
| `python3 -m holo net encode-want`           | Build control-plane WANT datagram.                      |
| `python3 -m holo net parse-control`         | Parse INVENTORY/WANT control datagram.                  |
| `python3 -m holo net assemble`              | Reassemble datagrams into complete chunks.              |
| `python3 -m holo net assembler-expire`      | Force expiration of partial reassembly state.           |
| `python3 -m holo net mesh-broadcast`        | `MeshNode.broadcast_chunk_dir` to peers.                |
| `python3 -m holo net mesh-recv`             | Loop `MeshNode.recv_once` and store chunks.             |
| `python3 -m holo net mesh-join-mcast`       | Join multicast groups (IPv4) on a socket.               |
| `python3 -m holo net mesh-inventory`        | `MeshNode.send_inventory` to peers.                     |
| `python3 -m holo net mesh-chunk-id`         | Extract chunk id from filename.                         |
| `python3 -m holo net mesh-recovery-id`      | Extract recovery id from filename.                      |
| `python3 -m holo net mesh-order-by-gain`    | Order chunk paths by gain metadata.                     |
| `python3 -m holo net mesh-local-chunk-ids`  | List local chunk ids for a content id.                  |
| `python3 -m holo net mesh-handle-inventory` | Handle INVENTORY and emit WANTs.                        |
| `python3 -m holo net mesh-handle-want`      | Handle WANT and send requested chunks.                  |
| `python3 -m holo net mesh-sendto`           | Send raw datagram bytes to a peer.                      |

</details>

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

## Field operations that matter operationally

Recovery (RLNC) is for heavy loss where graceful degradation is not enough. Systematic RLNC adds additional `recovery_*.holo` chunks that can reconstruct missing residual slices. Encoding uses `--recovery rlnc --overhead α`; decoding uses `--use-recovery`.

Healing is for maintaining a clean distribution of evidence after partial decodes, merges, or damage. It takes the current best-so-far reconstruction and re-encodes it back into a fresh chunk set, optionally iterating until changes stabilize (`--heal-fixed-point`) with drift guards for lossy v3.

Uncertainty output is about honesty: the decoder knows which residual coefficients were observed. `--write-uncertainty` writes confidence artifacts (images) or arrays (audio) where values near 1.0 mean fully observed.

Chunk priority is an efficiency knob. Encoders write per-chunk gain scores and a `manifest.json` ordering; you can decode “best K chunks” with `--prefer-gain`, and mesh senders can transmit high-gain chunks first.

Stacking is a field-native SNR trick: multiple noisy or partial exposures can be stacked by averaging reconstructions.

## Networking: URIs and content IDs

`holo://...` URIs map to a stable content identifier via `content_id = blake2s(holo://...)`. Above the codec, the net layer deals in datagrams plus a minimal control-plane (inventory/want). Loss and reordering are normal; the field representation is designed so it does not require retransmission to stay coherent.

## TNC audio/radio transport (experimental)

Holo does not turn images into audio content. The audio you transmit is just a modem carrier for bytes. The point of an audio/radio path is to move the same transport datagrams you would carry over UDP.

End-to-end signal flow:

```
image/audio -> holo.codec -> chunks (.holo)
  -> holo.net.transport (datagrams)
  -> holo.tnc (AFSK/PSK/FSK/OFDM modem)
  -> radio audio

radio audio
  -> holo.tnc -> datagrams -> chunks -> holo.codec decode
```

WAV size is dominated by bitrate. A crude rule of thumb is:

wav_bytes ~ payload_bytes * (16 * fs / baud)

Gaps and preambles add overhead. If you need smaller WAVs, raise `--baud`, lower `--fs`, reduce `--max-payload`, reduce chunk count, or use v3 with lower `--quality`.

One-line example commands (encode + TX, RX + decode):

```bash
# Noisy band (HF-like, v3): encode -> tnc-tx
PYTHONPATH=src python3 -m holo --olonomic src/flower.jpg --blocks 12 --quality 30 --recovery rlnc --overhead 0.25

PYTHONPATH=src python3 -m holo tnc-tx --chunk-dir src/flower.jpg.holo --uri holo://noise/demo --out tx_noise.wav \
  --max-payload 320 --gap-ms 40 --preamble-len 16 --fs 9600 --baud 1200 --prefer-gain --include-recovery

# Noisy band (HF-like): tnc-rx -> decode
PYTHONPATH=src python3 -m holo tnc-rx --input tx_noise.wav --uri holo://noise/demo --out rx_noise.holo --baud 1200 --preamble-len 16

PYTHONPATH=src python3 -m holo rx_noise.holo --output rx.png --use-recovery --prefer-gain
```

Suggested conservative modem parameters (AFSK) are summarized here:

| Link quality             | `--baud` | `--fs` | `--max-payload` | `--gap-ms` | `--include-recovery` | Compression/AGC |
| ------------------------ | -------- | ------ | --------------- | ---------- | -------------------- | --------------- |
| Noisy/variable (HF)      | 1200     | 9600   | 320             | 40         | yes                  | OFF             |
| Clean link (VHF/UHF/SHF) | 1200     | 9600   | 512             | 15         | optional             | OFF             |

If a WAV was edited/truncated and the header breaks, fix it with:

```bash
PYTHONPATH=src python3 -m holo tnc-wav-fix --input rx.wav --out rx_pcm.wav
```

If the file is badly corrupted, force raw decode:

```bash
PYTHONPATH=src python3 -m holo tnc-rx --input rx.wav --raw-s16le --raw-fs 48000 --out rx.holo
```

## HoloTV quickstart (experimental)

HoloTV schedules chunks across a window of frames and feeds them to a receiver:

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

## Olonomic v3: algorithm sketch

For images, v3 works by encoding a wave-domain residual. Operationally:

img -> coarse(img) -> coarse_up
residual = img - coarse_up
residual -> pad to block size -> per-block/channel DCT-II (ortho)
coeffs -> JPEG-style quantization (quality 1..100) -> zigzag order
coeff stream -> deterministic golden permutation -> slice across chunks -> zlib per chunk

Decoding inverts the above. Missing chunks are treated as missing coefficients (effectively zeroed), so the reconstruction loses detail rather than geometry. The final reconstruction is coarse + inverse-transformed residual.

For audio, the residual moves to an STFT domain:

audio -> coarse(audio) -> coarse_up
residual = audio - coarse_up
residual -> STFT (sqrt-Hann; default hop = n_fft/2)
bins -> frequency-shaped quant steps (quality 1..100) -> int16 (Re/Im interleaved)
stream -> golden permutation -> chunk slicing -> zlib

Metadata is carried inside the coarse payload (so chunk headers remain backward-compatible). For images this includes block size, quality, padding. For audio this includes n_fft, hop, and quant parameters.

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

<p align="center">
  <img width="800" alt="architecture map" src="https://github.com/user-attachments/assets/ca097bb5-3aaa-4efa-ba5b-8e6495cbae44" /><br/>
  <em>Codec → transport → field layering: genotype/phenotype/cortex/mesh analogy.</em>
</p>

## Testing

```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -p 'test_*.py'
```

Tests include round-trip checks, PSNR/MSE monotonicity for olonomic v3, and size-guard regressions.

A quick network/mesh smoke test (loopback UDP using `holo://` content IDs):

```bash
PYTHONPATH=src python3 src/examples/mesh_loopback.py
# emits galaxy.jpg chunks on 127.0.0.1, stores in src/examples/out/store_b/...,
# and writes a reconstructed image to src/examples/out/galaxy_mesh_recon.png
```

## Results snapshot

On `src/galaxy.jpg` with `coarse-side=16`, v3:

`python3 -m holo --olonomic src/galaxy.jpg --blocks 16 --quality 40 --packet-bytes 0`

Total size is ~0.35 MB (a coherent reconstruction is possible even from a single chunk). Under the same settings, v2 pixel residuals are ~1.69 MB. Visual quality is comparable, while v3 tends to degrade as “missing waves”, not missing pixels.

<p align="center">
  <img width="1280"  alt="photon collector" src="https://github.com/user-attachments/assets/c2b939d1-8911-4381-8bd7-a93e29f5401c" /><br/>
  <em>Photon-collector stacking: multiple exposures reinforce structure over noise.</em>
</p>

<p align="center">
  <img width="1280"  alt="psnr curves" src="https://github.com/user-attachments/assets/e8c700f2-e5b6-424b-a848-a230294e8269" /><br/>
  <em>PSNR vs received chunks: quality rises smoothly; variance stays low.</em>
</p>

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

