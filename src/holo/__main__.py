"""
holo.__main__

CLI entry point.

This file is intentionally small:
- parse args
- dispatch to the stable public API in holo.__init__
It must not contain codec math, storage policy, or networking logic.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image

import holo
from holo.codec import _write_wav_int16, save_image_rgb_u8


def _default_decode_output(in_dir: str, mode: str) -> str:
    base = in_dir[:-5] if in_dir.lower().endswith(".holo") else in_dir
    root, ext = os.path.splitext(base)

    if mode == "audio":
        return root + "_recon.wav"

    if not ext:
        ext = ".png"
    return root + "_recon" + ext


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo",
        description="HolographiX 3.0 holographic codec (v2 pixel residuals, v3 olonomic DCT/STFT): encode to .holo dirs, decode from .holo dirs.",
    )

    p.add_argument(
        "path",
        nargs="?",
        help="Input file (image or .wav) to encode, or .holo directory to decode. Not used when --stack is provided.",
    )
    p.add_argument("target_kb", nargs="?", type=int, help="Target chunk size in KB (encoding only). Example: 32")
    p.add_argument("-o", "--output", default=None, help="Output path for decoding (default: derived from dir name).")
    p.add_argument("--max-chunks", type=int, default=None, help="Decode using at most this many chunks (debug).")
    p.add_argument(
        "--prefer-gain",
        action="store_true",
        help="When decoding with --max-chunks, pick the best-K chunks by score if available.",
    )
    p.add_argument(
        "--write-uncertainty",
        action="store_true",
        help="Emit a confidence map/curve next to the reconstruction.",
    )
    p.add_argument(
        "--use-recovery",
        action="store_true",
        help="Use recovery chunks when present.",
    )
    p.add_argument(
        "--heal",
        action="store_true",
        help="Heal a .holo directory by re-encoding the best-so-far percept.",
    )
    p.add_argument(
        "--heal-fixed-point",
        action="store_true",
        help="Iterative healing until convergence (fixed-point).",
    )
    p.add_argument(
        "--heal-out",
        default=None,
        help="Output .holo directory for healing (default: derived from input).",
    )
    p.add_argument(
        "--heal-target-kb",
        type=int,
        default=32,
        help="Target chunk size in KB for healing output.",
    )
    p.add_argument(
        "--heal-iters",
        type=int,
        default=4,
        help="Max iterations for fixed-point healing.",
    )
    p.add_argument(
        "--heal-tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for fixed-point healing.",
    )
    p.add_argument(
        "--heal-metric",
        choices=["mse", "psnr", "l2"],
        default="mse",
        help="Convergence metric for fixed-point healing.",
    )
    p.add_argument(
        "--heal-honest",
        action="store_true",
        help="Attenuate uncertain regions during healing.",
    )
    p.add_argument(
        "--stack",
        nargs="+",
        default=None,
        help="Stack multiple .holo directories into one reconstruction (pass 2+ .holo paths).",
    )
    p.add_argument(
        "--stack-max-chunks",
        type=int,
        default=None,
        help="Limit chunks per exposure when stacking.",
    )

    p.add_argument("--blocks", type=int, default=None, help="Explicit chunk count for encoding.")
    p.add_argument("--coarse-side", type=int, default=16, help="Image coarse max side (encoding).")
    p.add_argument(
        "--coarse-format",
        choices=["png", "jpeg", "webp"],
        default="png",
        help="Coarse thumbnail format (encoding).",
    )
    p.add_argument(
        "--coarse-quality",
        type=int,
        default=None,
        help="Coarse thumbnail quality for lossy formats (encoding).",
    )
    p.add_argument("--coarse-frames", type=int, default=2048, help="Audio coarse max frames (encoding).")
    p.add_argument(
        "--coarse-model",
        choices=["downsample", "latent_lowfreq", "ae_latent"],
        default="downsample",
        help="Coarse model for olonomic (v3) encoding.",
    )
    p.add_argument(
        "--packet-bytes",
        type=int,
        default=0,
        help="Approximate per-chunk byte budget to stay within one UDP datagram (set >0 to enable; default 0 = no limit).",
    )
    p.add_argument(
        "--olonomic",
        action="store_true",
        help="Use olonomic (version 3) codec for encoding.",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=50,
        help="Residual quantization quality for olonomic (v3) codec (1-100, higher = better quality / larger).",
    )
    p.add_argument(
        "--recovery",
        choices=["rlnc"],
        default=None,
        help="Enable recovery chunks (systematic RLNC).",
    )
    p.add_argument(
        "--overhead",
        type=float,
        default=0.0,
        help="Recovery overhead fraction (e.g. 0.25 = +25% chunks).",
    )
    p.add_argument(
        "--recovery-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic recovery coefficients.",
    )

    return p.parse_args(argv)


def _parse_tnc_tx_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo tnc-tx",
        description="Encode a .holo chunk dir into an AFSK WAV file.",
    )
    p.add_argument("--chunk-dir", required=True, help="Input .holo directory with chunk_*.holo files.")
    p.add_argument("--uri", default=None, help="Content URI (used to derive content_id).")
    p.add_argument("--out", default="tx_afsk.wav", help="Output WAV file path.")
    p.add_argument("--max-payload", type=int, default=512, help="Datagram payload size budget in bytes.")
    p.add_argument("--gap-ms", type=float, default=20.0, help="Silence gap between datagrams (ms).")
    p.add_argument("--prefer-gain", action="store_true", help="Send higher-gain chunks first when manifest/meta is present.")
    p.add_argument("--include-recovery", action="store_true", help="Include recovery_*.holo chunks if present.")
    p.add_argument("--max-chunks", type=int, default=None, help="Limit number of chunks to send (debug/preview).")
    p.add_argument("--fs", type=int, default=48000, help="Sample rate for WAV output (Hz).")
    p.add_argument("--baud", type=int, default=1200, help="AFSK baud rate.")
    p.add_argument("--preamble-len", type=int, default=16, help="AFSK frame preamble length (bytes).")
    return p.parse_args(argv)


def _parse_tnc_rx_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo tnc-rx",
        description="Decode an AFSK WAV file into a .holo chunk dir.",
    )
    p.add_argument("--input", required=True, help="Input WAV file (recorded from radio).")
    p.add_argument("--out", required=True, help="Output directory for reconstructed chunks.")
    p.add_argument("--uri", default=None, help="Optional content URI to filter by.")
    p.add_argument("--baud", type=int, default=1200, help="AFSK baud rate.")
    p.add_argument("--preamble-len", type=int, default=16, help="AFSK frame preamble length (bytes).")
    p.add_argument("--raw-s16le", action="store_true", help="Treat input as raw PCM s16le (ignore WAV header).")
    p.add_argument("--raw-fs", type=int, default=None, help="Sample rate for raw PCM input.")
    p.add_argument("--raw-ch", type=int, default=1, help="Channels for raw PCM input.")
    p.add_argument("--raw-skip", type=int, default=None, help="Bytes to skip for raw PCM input.")
    p.add_argument("--force-pcm16", action="store_true", default=True, help="Force best-effort PCM16 decode (default: on).")
    p.add_argument("--no-force-pcm16", dest="force_pcm16", action="store_false", help="Disable forced PCM16 decode.")
    return p.parse_args(argv)


def _parse_tnc_wav_fix_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo tnc-wav-fix",
        description="Re-encode a WAV (or raw s16le) into PCM16 mono.",
    )
    p.add_argument("--input", required=True, help="Input WAV file (or raw PCM if --raw-s16le).")
    p.add_argument("--out", required=True, help="Output PCM16 WAV path.")
    p.add_argument("--fs", type=int, default=None, help="Output sample rate (Hz).")
    p.add_argument("--raw-s16le", action="store_true", help="Treat input as raw PCM s16le (ignore WAV header).")
    p.add_argument("--raw-fs", type=int, default=None, help="Sample rate for raw PCM input.")
    p.add_argument("--raw-ch", type=int, default=1, help="Channels for raw PCM input.")
    p.add_argument("--raw-skip", type=int, default=None, help="Bytes to skip for raw PCM input.")
    return p.parse_args(argv)


def _default_uri_from_path(path: str, *, prefix: str = "holo://radio/") -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in base)
    if not safe:
        safe = "default"
    return prefix + safe


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] in {"tnc-tx", "tnc-rx", "tnc-wav-fix"}:
        from holo.tnc.cli import decode_wav_to_chunk_dir, encode_chunk_dir_to_wav, fix_wav_to_pcm

        cmd = argv[0]
        if cmd == "tnc-tx":
            args = _parse_tnc_tx_args(argv[1:])
            uri = args.uri or _default_uri_from_path(args.chunk_dir, prefix="holo://radio/")
            encode_chunk_dir_to_wav(
                args.chunk_dir,
                args.out,
                uri=uri,
                max_payload=int(args.max_payload),
                gap_ms=float(args.gap_ms),
                prefer_gain=bool(args.prefer_gain),
                include_recovery=bool(args.include_recovery),
                max_chunks=args.max_chunks,
                fs=int(args.fs),
                baud=int(args.baud),
                preamble_len=int(args.preamble_len),
            )
            print(args.out)
            if args.uri is None:
                print(f"URI: {uri}")
            return 0
        if cmd == "tnc-rx":
            args = _parse_tnc_rx_args(argv[1:])
            decode_wav_to_chunk_dir(
                args.input,
                args.out,
                uri=args.uri,
                baud=int(args.baud),
                preamble_len=int(args.preamble_len),
                raw_s16le=bool(args.raw_s16le),
                raw_fs=args.raw_fs,
                raw_channels=int(args.raw_ch),
                raw_skip=args.raw_skip,
                force_pcm16=bool(args.force_pcm16),
            )
            print(args.out)
            return 0
        if cmd == "tnc-wav-fix":
            args = _parse_tnc_wav_fix_args(argv[1:])
            fix_wav_to_pcm(
                args.input,
                args.out,
                fs=args.fs,
                raw_s16le=bool(args.raw_s16le),
                raw_fs=args.raw_fs,
                raw_channels=int(args.raw_ch),
                raw_skip=args.raw_skip,
            )
            print(args.out)
            return 0

    args = _parse_args(argv)

    # Stacking mode (no direct CLI subcommand before; invoked with --stack)
    if args.stack:
        if len(args.stack) < 2:
            raise ValueError("--stack requires at least two .holo directories")
        out = args.output or "stacked_recon.png"
        holo.stack_image_holo_dirs(args.stack, out, max_chunks=args.stack_max_chunks)
        print(out)
        return 0

    path = os.path.normpath(args.path)

    if os.path.isdir(path) and path.lower().endswith(".holo"):
        if args.heal or args.heal_fixed_point:
            from holo.field import Field

            base = path[:-5] if path.lower().endswith(".holo") else path
            if args.heal_out:
                out_dir = args.heal_out
            elif args.output:
                out_dir = args.output
            elif args.heal_fixed_point:
                out_dir = base + "_healed_fp.holo"
            else:
                out_dir = base + "_healed.holo"

            field = Field(content_id="cli/heal", chunk_dir=path)
            if args.heal_fixed_point:
                report = field.heal_fixed_point(
                    out_dir,
                    max_iters=int(args.heal_iters),
                    tol=float(args.heal_tol),
                    metric=str(args.heal_metric),
                    target_chunk_kb=int(args.heal_target_kb),
                    max_chunks=args.max_chunks,
                    honest=bool(args.heal_honest),
                    prefer_gain=bool(args.prefer_gain),
                    use_recovery=True if args.use_recovery else None,
                )
                print(report.get("out_dir", out_dir))
                print(report)
            else:
                field.heal_once(
                    out_dir,
                    target_chunk_kb=int(args.heal_target_kb),
                    max_chunks=args.max_chunks,
                    honest=bool(args.heal_honest),
                    prefer_gain=bool(args.prefer_gain),
                    use_recovery=True if args.use_recovery else None,
                )
                print(out_dir)
            return 0

        mode = holo.detect_mode_from_chunk_dir(path)
        out = args.output or _default_decode_output(path, mode)
        use_recovery = True if args.use_recovery else None

        if args.write_uncertainty:
            if mode == "audio":
                audio, sr, conf = holo.decode_audio_holo_dir_meta(
                    path,
                    max_chunks=args.max_chunks,
                    prefer_gain=args.prefer_gain,
                    use_recovery=use_recovery,
                    return_sr=True,
                )
                _write_wav_int16(out, audio, int(sr))
                root, _ext = os.path.splitext(out)
                conf_path = root + "_confidence.npy"
                np.save(conf_path, conf.astype(np.float32))
            else:
                recon, conf = holo.decode_image_holo_dir_meta(
                    path,
                    max_chunks=args.max_chunks,
                    prefer_gain=args.prefer_gain,
                    use_recovery=use_recovery,
                )
                save_image_rgb_u8(recon, out)
                root, _ext = os.path.splitext(out)
                conf_path = root + "_confidence.png"
                conf_u8 = np.clip(conf * 255.0, 0.0, 255.0).astype(np.uint8)
                Image.fromarray(conf_u8, mode="L").save(conf_path)
        else:
            if mode == "audio":
                holo.decode_audio_holo_dir(
                    path,
                    out,
                    max_chunks=args.max_chunks,
                    prefer_gain=args.prefer_gain,
                    use_recovery=use_recovery,
                )
            else:
                holo.decode_image_holo_dir(
                    path,
                    out,
                    max_chunks=args.max_chunks,
                    prefer_gain=args.prefer_gain,
                    use_recovery=use_recovery,
                )

        print(out)
        if args.write_uncertainty:
            print(conf_path)
        return 0

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mode = holo.detect_mode_from_extension(path)
    out_dir = path + ".holo"

    if mode == "audio":
        encode_kwargs = {}
        if args.target_kb is not None:
            encode_kwargs["target_chunk_kb"] = int(args.target_kb)
        if args.blocks is not None:
            encode_kwargs["block_count"] = int(args.blocks)
        encode_kwargs.setdefault("coarse_max_frames", int(args.coarse_frames))
        if args.olonomic:
            encode_kwargs.setdefault("quality", int(args.quality))
            encode_kwargs.setdefault("coarse_model", str(args.coarse_model))
            if args.recovery:
                encode_kwargs["recovery"] = str(args.recovery)
                encode_kwargs["overhead"] = float(args.overhead)
                if args.recovery_seed is not None:
                    encode_kwargs["recovery_seed"] = int(args.recovery_seed)
            holo.encode_audio_olonomic_holo_dir(path, out_dir, **encode_kwargs)
        else:
            holo.encode_audio_holo_dir(path, out_dir, **encode_kwargs)
    else:
        encode_kwargs = {}
        if args.target_kb is not None:
            encode_kwargs["target_chunk_kb"] = int(args.target_kb)
        if args.blocks is not None:
            encode_kwargs["block_count"] = int(args.blocks)
        if args.packet_bytes and args.packet_bytes > 0:
            encode_kwargs["max_chunk_bytes"] = int(args.packet_bytes)
        encode_kwargs.setdefault("coarse_max_side", int(args.coarse_side))
        if args.olonomic:
            encode_kwargs.setdefault("quality", int(args.quality))
            encode_kwargs.setdefault("coarse_model", str(args.coarse_model))
            if args.recovery:
                encode_kwargs["recovery"] = str(args.recovery)
                encode_kwargs["overhead"] = float(args.overhead)
                if args.recovery_seed is not None:
                    encode_kwargs["recovery_seed"] = int(args.recovery_seed)
            holo.encode_image_olonomic_holo_dir(path, out_dir, **encode_kwargs)
        else:
            encode_kwargs.setdefault("coarse_format", args.coarse_format.upper())
            if args.coarse_quality is not None:
                encode_kwargs["coarse_quality"] = int(args.coarse_quality)
            holo.encode_image_holo_dir(path, out_dir, **encode_kwargs)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
