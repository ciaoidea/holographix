#!/usr/bin/env python3
"""
Train a tiny linear autoencoder and export weights for ae_latent.

This script is optional and not used in tests. It requires torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_array(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim < 2:
        raise ValueError("Expected an array with leading batch dimension")
    return arr


def _train_linear_ae(x: np.ndarray, latent: int, epochs: int, lr: float) -> dict:
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("torch is required to run this script") from exc

    device = torch.device("cpu")
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    in_dim = x_t.shape[1]

    encoder = nn.Linear(in_dim, latent, bias=True)
    decoder = nn.Linear(latent, in_dim, bias=True)
    model = nn.Sequential(encoder, nn.Tanh(), decoder)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(int(epochs)):
        opt.zero_grad()
        out = model(x_t)
        loss = loss_fn(out, x_t)
        loss.backward()
        opt.step()

    return {
        "enc_w": encoder.weight.detach().cpu().numpy(),
        "enc_b": encoder.bias.detach().cpu().numpy(),
        "dec_w": decoder.weight.detach().cpu().numpy(),
        "dec_b": decoder.bias.detach().cpu().numpy(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny linear AE and export weights as .npz.")
    ap.add_argument("--mode", choices=["image", "audio"], default="image", help="Input modality.")
    ap.add_argument("--input", required=True, help="Path to .npy training data (batch-first).")
    ap.add_argument("--latent", type=int, default=32, help="Latent size.")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--output", required=True, help="Output .npz path.")
    args = ap.parse_args()

    data = _load_array(Path(args.input))
    if args.mode == "image":
        if data.ndim != 4:
            raise ValueError("Image data must be N x H x W x C")
        n, h, w, c = data.shape
        x = data.reshape(n, -1).astype(np.float32) / 255.0
        input_shape = (h, w, c)
    else:
        if data.ndim != 2:
            raise ValueError("Audio data must be N x T (mono)")
        n, t = data.shape
        x = data.astype(np.float32) / 32768.0
        input_shape = (t,)

    weights = _train_linear_ae(x, int(args.latent), int(args.epochs), float(args.lr))
    weights["input_shape"] = np.array(input_shape, dtype=np.int64)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **weights)


if __name__ == "__main__":
    main()
