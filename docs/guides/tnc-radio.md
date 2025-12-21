# TNC Radio Guide (AFSK WAV)

This guide uses the built-in AFSK modem to carry holographic datagrams over
an audio channel (WAV files or radio links).

## 1) Encode a .holo Directory to WAV

```bash
python3 -m holo tnc-tx \
  --chunk-dir src/flower.jpg.holo \
  --uri holo://demo/flower \
  --out tx_afsk.wav \
  --max-payload 512 \
  --baud 1200 \
  --fs 48000
```

Optional flags:
- `--prefer-gain`        Send higher-gain chunks first (if manifest/meta exists).
- `--include-recovery`   Include `recovery_*.holo` chunks.
- `--max-chunks N`       Limit the number of chunks (debug/preview).
- `--preamble-len N`     Frame preamble length in bytes.

## 2) Decode WAV Back to Chunks

```bash
python3 -m holo tnc-rx \
  --input rx.wav \
  --out rx_chunks \
  --uri holo://demo/flower
```

If the WAV is not valid PCM16, first normalize it:

```bash
python3 -m holo tnc-wav-fix --input rx.wav --out rx_fixed.wav
python3 -m holo tnc-rx --input rx_fixed.wav --out rx_chunks --uri holo://demo/flower
```

For raw PCM input:

```bash
python3 -m holo tnc-rx --input rx.raw --out rx_chunks --raw-s16le --raw-fs 48000
```

## 3) Decode the Reconstruction

```bash
python3 -m holo rx_chunks/<content-id-hex> --output recon.png
```

## Modem Details (Current Defaults)
- AFSK 1200 baud
- Mark = 1200 Hz, Space = 2200 Hz
- Preamble byte = 0x55, sync word = 0x2d 0xd4
- CRC16-CCITT on each frame

See `src/holo/tnc/afsk.py` and `src/holo/tnc/frame.py` for exact behavior.
