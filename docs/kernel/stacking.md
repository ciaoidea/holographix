# Gauge Alignment in Stacking (v3)

This note defines how stacking aligns v3 (olonomic) fields before summation.
The goal is to superpose evidence in the same gauge so coefficient-domain
averaging remains coherent.

## Scope

Applies only to v3 fields (image DCT blocks, audio STFT frames). If any input
is not v3, stacking falls back to a simple pixel/sample average.

## Images (v3)

1. Build a coarse block map from the v3 coarse base:
   - Convert coarse RGB to grayscale.
   - Pad to block grid.
   - Average-pool into (blocks_h, blocks_w).
   - Remove DC and apply a Hanning window to stabilize correlation.
2. Estimate an integer block shift (dy, dx) by phase correlation of the block
   maps.
3. Apply the shift to the coefficient tensor on the block grid:
   - Shift only whole blocks (exact permutation in this codec).
   - Zero-fill out-of-frame blocks.
   - Shift the coefficient mask (or derive a geometric mask if none exists).
4. Shift the coarse base by (dy * block_size, dx * block_size) in pixels and
   track a geometric validity mask so zeros do not bias the average.
5. Average coefficients where masks indicate valid coverage; average coarse
   using the geometric mask.

If correlation is weak (peak below a threshold) or fails, the shift is treated
as (0, 0) and stacking proceeds without alignment.

## Audio (v3)

1. Build a low-rate envelope from the coarse waveform:
   - For each STFT frame, compute mean absolute amplitude over the frame window.
   - Remove DC and apply a Hanning window.
2. Estimate an integer frame shift by phase correlation of the envelopes.
3. Shift the coefficient tensor by whole STFT frames (zero-fill out-of-frame).
4. Shift the coarse waveform by (shift_frames * hop) samples and average using
   a geometric validity mask.

If correlation is weak or fails, the shift is treated as 0.

## Usage

Stacking auto-detects image vs audio from the input directories:

```bash
python3 -m holo --stack a.holo b.holo --output stacked_recon.png
python3 -m holo --stack a_audio.holo b_audio.holo --output stacked_recon.wav
```

Disable gauge alignment (pure averaging):

```bash
python3 -m holo --stack a.holo b.holo --stack-no-gauge
```

## Limitations

- Image alignment is limited to integer block shifts (block_size units).
  Sub-block shifts would require a transform that supports phase ramps
  (e.g., complex FFT), which v3 image DCT blocks do not.
- Audio alignment is limited to integer frame shifts (hop units).
- Gauge alignment is best-effort and is skipped when correlation is weak.
