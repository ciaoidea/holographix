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
import base64
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

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


def _parse_hex_bytes(value: str) -> bytes:
    if value is None:
        raise ValueError("missing hex value")
    cleaned = value.strip()
    if cleaned.startswith(("0x", "0X")):
        cleaned = cleaned[2:]
    return bytes.fromhex(cleaned)


def _encode_bytes(data: bytes, fmt: str) -> str:
    if fmt == "hex":
        return data.hex()
    if fmt == "base64":
        return base64.b64encode(data).decode("ascii")
    raise ValueError(f"unsupported format: {fmt}")


def _emit_bytes(data: bytes, fmt: str, out_path: Optional[str]) -> None:
    if fmt == "raw":
        if out_path:
            Path(out_path).write_bytes(data)
        else:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        return
    text = _encode_bytes(data, fmt)
    if out_path:
        Path(out_path).write_text(text + "\n", encoding="ascii")
    else:
        print(text)


def _parse_addr(addr: str) -> tuple[str, int]:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _read_input_bytes(*, hex_str: Optional[str], b64_str: Optional[str], path: Optional[str]) -> bytes:
    provided = [hex_str is not None, b64_str is not None, path is not None]
    if sum(provided) != 1:
        raise ValueError("provide exactly one of --data-hex, --data-base64, or --input")
    if path is not None:
        return Path(path).read_bytes()
    if hex_str is not None:
        return _parse_hex_bytes(hex_str)
    return base64.b64decode(b64_str.encode("ascii"))


def _iter_encoded_lines(path: Optional[str], fmt: str) -> Iterable[bytes]:
    text = Path(path).read_text(encoding="ascii") if path else sys.stdin.read()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if fmt == "hex":
            yield _parse_hex_bytes(line)
        elif fmt == "base64":
            yield base64.b64decode(line.encode("ascii"))
        else:
            raise ValueError(f"unsupported line format: {fmt}")


def _content_id_from_args(args: argparse.Namespace) -> bytes:
    from holo.net.arch import content_id_bytes_from_uri

    if getattr(args, "uri", None):
        return content_id_bytes_from_uri(args.uri, digest_size=int(args.digest_size))
    if getattr(args, "content_id", None):
        cid = _parse_hex_bytes(args.content_id)
        if len(cid) != 16:
            raise ValueError("content_id must be 16 bytes (32 hex chars)")
        return cid
    raise SystemExit("Provide --uri or --content-id.")


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


def _parse_net_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m holo net",
        description="holo.net/holo.net.transport CLI helpers.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_const = sub.add_parser("constants", help="Print transport constants.")

    p_norm = sub.add_parser("norm-uri", help="Normalize a holo:// URI.")
    p_norm.add_argument("--uri", required=True)

    p_id = sub.add_parser("id-bytes", help="content_id_bytes_from_uri")
    p_id.add_argument("--uri", required=True)
    p_id.add_argument("--digest-size", type=int, default=16)
    p_id.add_argument("--format", choices=["hex", "base64", "raw"], default="hex")
    p_id.add_argument("--out", default=None)

    p_id_hex = sub.add_parser("id-hex", help="content_id_hex_from_uri")
    p_id_hex.add_argument("--uri", required=True)
    p_id_hex.add_argument("--digest-size", type=int, default=16)
    p_id_hex.add_argument("--out", default=None)

    p_stream = sub.add_parser("stream-id", help="content_id_bytes_from_stream_frame")
    p_stream.add_argument("--stream-id", required=True)
    p_stream.add_argument("--frame-idx", required=True, type=int)
    p_stream.add_argument("--digest-size", type=int, default=16)
    p_stream.add_argument("--format", choices=["hex", "base64", "raw"], default="hex")
    p_stream.add_argument("--out", default=None)

    p_iter = sub.add_parser("iter-datagrams", help="iter_chunk_datagrams")
    p_iter.add_argument("--uri", default=None)
    p_iter.add_argument("--content-id", default=None)
    p_iter.add_argument("--digest-size", type=int, default=16)
    p_iter.add_argument("--chunk-id", type=int, required=True)
    p_iter.add_argument("--chunk-file", required=True)
    p_iter.add_argument("--max-payload", type=int, default=1200)
    p_iter.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_iter.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_iter.add_argument("--enc-key-id", type=int, default=0)
    p_iter.add_argument("--format", choices=["hex", "base64"], default="hex")
    p_iter.add_argument("--out", default=None, help="Write encoded datagrams (one per line).")
    p_iter.add_argument("--out-dir", default=None, help="Write raw datagram bytes to this dir.")

    p_send = sub.add_parser("send-chunk", help="send_chunk")
    p_send.add_argument("--addr", required=True, help="ip:port")
    p_send.add_argument("--bind", default=None, help="optional local bind ip:port")
    p_send.add_argument("--uri", default=None)
    p_send.add_argument("--content-id", default=None)
    p_send.add_argument("--digest-size", type=int, default=16)
    p_send.add_argument("--chunk-id", type=int, required=True)
    p_send.add_argument("--chunk-file", required=True)
    p_send.add_argument("--max-payload", type=int, default=1200)
    p_send.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_send.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_send.add_argument("--enc-key-id", type=int, default=0)

    p_inv = sub.add_parser("encode-inventory", help="encode_inventory_datagram")
    p_inv.add_argument("--uri", default=None)
    p_inv.add_argument("--content-id", default=None)
    p_inv.add_argument("--digest-size", type=int, default=16)
    p_inv.add_argument("--chunk-ids", nargs="+", type=int, required=True)
    p_inv.add_argument("--format", choices=["hex", "base64", "raw"], default="hex")
    p_inv.add_argument("--out", default=None)

    p_want = sub.add_parser("encode-want", help="encode_want_datagram")
    p_want.add_argument("--uri", default=None)
    p_want.add_argument("--content-id", default=None)
    p_want.add_argument("--digest-size", type=int, default=16)
    p_want.add_argument("--chunk-ids", nargs="+", type=int, required=True)
    p_want.add_argument("--format", choices=["hex", "base64", "raw"], default="hex")
    p_want.add_argument("--out", default=None)

    p_parse = sub.add_parser("parse-control", help="parse_control_datagram")
    p_parse_group = p_parse.add_mutually_exclusive_group(required=True)
    p_parse_group.add_argument("--data-hex", default=None)
    p_parse_group.add_argument("--data-base64", default=None)
    p_parse_group.add_argument("--input", default=None, help="Read raw datagram bytes from file.")

    p_asm = sub.add_parser("assemble", help="ChunkAssembler over encoded datagram lines.")
    p_asm.add_argument("--input", default=None, help="Read encoded datagram lines from file (default: stdin).")
    p_asm.add_argument("--format", choices=["hex", "base64"], default="hex")
    p_asm.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_asm.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_asm.add_argument("--enc-key-id", type=int, default=0)
    p_asm.add_argument("--max-partial-age", type=float, default=5.0)
    p_asm.add_argument("--out", default=None, help="Write first completed chunk to this file.")
    p_asm.add_argument("--out-dir", default=None, help="Write completed chunks under this dir.")

    p_exp = sub.add_parser("assembler-expire", help="ChunkAssembler._expire_partials")
    p_exp.add_argument("--input", default=None, help="Read encoded datagram lines from file (default: stdin).")
    p_exp.add_argument("--format", choices=["hex", "base64"], default="hex")
    p_exp.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_exp.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_exp.add_argument("--enc-key-id", type=int, default=0)
    p_exp.add_argument("--max-partial-age", type=float, default=5.0)
    p_exp.add_argument("--advance", type=float, default=0.0, help="Seconds to advance time before expiring.")

    p_mesh_b = sub.add_parser("mesh-broadcast", help="MeshNode.broadcast_chunk_dir")
    p_mesh_b.add_argument("--uri", required=True)
    p_mesh_b.add_argument("--chunk-dir", required=True)
    p_mesh_b.add_argument("--peer", action="append", default=[], help="peer ip:port (repeatable)")
    p_mesh_b.add_argument("--bind", default=None, help="optional local bind ip:port")
    p_mesh_b.add_argument("--store-root", default="cortex_tx")
    p_mesh_b.add_argument("--repeats", type=int, default=1)
    p_mesh_b.add_argument("--priority", choices=["gain"], default=None)
    p_mesh_b.add_argument("--send-recovery", action="store_true")
    p_mesh_b.add_argument("--max-payload", type=int, default=1200)
    p_mesh_b.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_mesh_b.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_mesh_b.add_argument("--enc-key-id", type=int, default=0)

    p_mesh_r = sub.add_parser("mesh-recv", help="MeshNode.recv_once loop")
    p_mesh_r.add_argument("--bind", required=True, help="ip:port to bind")
    p_mesh_r.add_argument("--store-root", default="cortex_rx")
    p_mesh_r.add_argument("--duration", type=float, default=5.0)
    p_mesh_r.add_argument("--max-payload", type=int, default=1200)
    p_mesh_r.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_mesh_r.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_mesh_r.add_argument("--enc-key-id", type=int, default=0)
    p_mesh_r.add_argument("--mcast", action="append", default=[], help="multicast group ip:port (repeatable)")

    p_mesh_join = sub.add_parser("mesh-join-mcast", help="MeshNode._join_mcast_groups")
    p_mesh_join.add_argument("--mcast", action="append", default=[], required=True, help="multicast group ip:port")
    p_mesh_join.add_argument("--duration", type=float, default=1.0, help="Seconds to keep socket open.")

    p_mesh_inv = sub.add_parser("mesh-inventory", help="MeshNode.send_inventory")
    p_mesh_inv.add_argument("--uri", default=None)
    p_mesh_inv.add_argument("--content-id", default=None)
    p_mesh_inv.add_argument("--digest-size", type=int, default=16)
    p_mesh_inv.add_argument("--peer", action="append", default=[], help="peer ip:port (repeatable)")
    p_mesh_inv.add_argument("--bind", default=None)
    p_mesh_inv.add_argument("--store-root", default="cortex_node")
    p_mesh_inv.add_argument("--max-payload", type=int, default=1200)
    p_mesh_inv.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_mesh_inv.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_mesh_inv.add_argument("--enc-key-id", type=int, default=0)

    p_mesh_cid = sub.add_parser("mesh-chunk-id", help="MeshNode._chunk_id_from_name")
    p_mesh_cid.add_argument("--path", required=True)

    p_mesh_rid = sub.add_parser("mesh-recovery-id", help="MeshNode._recovery_id_from_name")
    p_mesh_rid.add_argument("--path", required=True)

    p_mesh_ord = sub.add_parser("mesh-order-by-gain", help="MeshNode._ordered_by_gain")
    p_mesh_ord.add_argument("--chunk-dir", required=True)
    p_mesh_ord.add_argument("--store-root", default="cortex_node")
    p_mesh_ord.add_argument("--max-payload", type=int, default=1200)

    p_mesh_local = sub.add_parser("mesh-local-chunk-ids", help="MeshNode._local_chunk_ids")
    p_mesh_local.add_argument("--uri", default=None)
    p_mesh_local.add_argument("--content-id", default=None)
    p_mesh_local.add_argument("--digest-size", type=int, default=16)
    p_mesh_local.add_argument("--store-root", default="cortex_node")
    p_mesh_local.add_argument("--max-payload", type=int, default=1200)

    p_mesh_hi = sub.add_parser("mesh-handle-inventory", help="MeshNode._handle_inventory")
    p_mesh_hi.add_argument("--peer", required=True, help="ip:port")
    p_mesh_hi.add_argument("--uri", default=None)
    p_mesh_hi.add_argument("--content-id", default=None)
    p_mesh_hi.add_argument("--digest-size", type=int, default=16)
    p_mesh_hi.add_argument("--chunk-ids", nargs="+", type=int, required=True)
    p_mesh_hi.add_argument("--store-root", default="cortex_node")
    p_mesh_hi.add_argument("--max-payload", type=int, default=1200)
    p_mesh_hi.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_mesh_hi.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_mesh_hi.add_argument("--enc-key-id", type=int, default=0)

    p_mesh_hw = sub.add_parser("mesh-handle-want", help="MeshNode._handle_want")
    p_mesh_hw.add_argument("--peer", required=True, help="ip:port")
    p_mesh_hw.add_argument("--uri", default=None)
    p_mesh_hw.add_argument("--content-id", default=None)
    p_mesh_hw.add_argument("--digest-size", type=int, default=16)
    p_mesh_hw.add_argument("--chunk-ids", nargs="+", type=int, required=True)
    p_mesh_hw.add_argument("--store-root", default="cortex_node")
    p_mesh_hw.add_argument("--max-payload", type=int, default=1200)
    p_mesh_hw.add_argument("--auth-key", default=None, help="Hex HMAC key")
    p_mesh_hw.add_argument("--enc-key", default=None, help="Hex AES-GCM key")
    p_mesh_hw.add_argument("--enc-key-id", type=int, default=0)

    p_mesh_sendto = sub.add_parser("mesh-sendto", help="MeshNode._sendto")
    p_mesh_sendto.add_argument("--peer", required=True, help="ip:port")
    p_mesh_sendto_group = p_mesh_sendto.add_mutually_exclusive_group(required=True)
    p_mesh_sendto_group.add_argument("--data-hex", default=None)
    p_mesh_sendto_group.add_argument("--data-base64", default=None)
    p_mesh_sendto_group.add_argument("--input", default=None, help="Read raw datagram bytes from file.")

    return p.parse_args(argv)


def _default_uri_from_path(path: str, *, prefix: str = "holo://radio/") -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in base)
    if not safe:
        safe = "default"
    return prefix + safe


def _run_net(argv: Optional[list[str]] = None) -> int:
    args = _parse_net_args(argv)

    if args.cmd == "constants":
        from holo.net import transport as net_transport

        data = {
            "MAGIC": net_transport.MAGIC.hex(),
            "CONTROL_MAGIC": net_transport.CONTROL_MAGIC.hex(),
            "CTRL_TYPE_INV": int(net_transport.CTRL_TYPE_INV),
            "CTRL_TYPE_WANT": int(net_transport.CTRL_TYPE_WANT),
            "MAX_CTRL_CHUNKS": int(net_transport.MAX_CTRL_CHUNKS),
        }
        print(json.dumps(data))
        return 0

    if args.cmd == "norm-uri":
        from holo.net.arch import _norm_holo_uri

        print(_norm_holo_uri(args.uri))
        return 0

    if args.cmd == "id-bytes":
        from holo.net.arch import content_id_bytes_from_uri

        cid = content_id_bytes_from_uri(args.uri, digest_size=int(args.digest_size))
        _emit_bytes(cid, args.format, args.out)
        return 0

    if args.cmd == "id-hex":
        from holo.net.arch import content_id_hex_from_uri

        cid = content_id_hex_from_uri(args.uri, digest_size=int(args.digest_size))
        if args.out:
            Path(args.out).write_text(cid + "\n", encoding="ascii")
        else:
            print(cid)
        return 0

    if args.cmd == "stream-id":
        from holo.net.arch import content_id_bytes_from_stream_frame

        cid = content_id_bytes_from_stream_frame(args.stream_id, int(args.frame_idx), digest_size=int(args.digest_size))
        _emit_bytes(cid, args.format, args.out)
        return 0

    if args.cmd == "iter-datagrams":
        from holo.net.transport import iter_chunk_datagrams

        if args.out and args.out_dir:
            raise SystemExit("Use --out or --out-dir, not both.")
        content_id = _content_id_from_args(args)
        chunk_bytes = Path(args.chunk_file).read_bytes()
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        datagrams = list(
            iter_chunk_datagrams(
                content_id,
                int(args.chunk_id),
                chunk_bytes,
                max_payload=int(args.max_payload),
                auth_key=auth_key,
                enc_key=enc_key,
                key_id=int(args.enc_key_id),
            )
        )
        if args.out_dir:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, dg in enumerate(datagrams):
                (out_dir / f"datagram_{idx:04d}.bin").write_bytes(dg)
        else:
            lines = [_encode_bytes(dg, args.format) for dg in datagrams]
            text = "\n".join(lines) + ("\n" if lines else "")
            if args.out:
                Path(args.out).write_text(text, encoding="ascii")
            else:
                sys.stdout.write(text)
        return 0

    if args.cmd == "send-chunk":
        from holo.net.transport import send_chunk

        content_id = _content_id_from_args(args)
        chunk_bytes = Path(args.chunk_file).read_bytes()
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if args.bind:
            sock.bind(_parse_addr(args.bind))
        send_chunk(
            sock,
            _parse_addr(args.addr),
            content_id,
            int(args.chunk_id),
            chunk_bytes,
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            key_id=int(args.enc_key_id),
        )
        print(f"sent chunk {int(args.chunk_id)} to {args.addr}")
        return 0

    if args.cmd == "encode-inventory":
        from holo.net.transport import encode_inventory_datagram

        content_id = _content_id_from_args(args)
        data = encode_inventory_datagram(content_id, args.chunk_ids)
        _emit_bytes(data, args.format, args.out)
        return 0

    if args.cmd == "encode-want":
        from holo.net.transport import encode_want_datagram

        content_id = _content_id_from_args(args)
        data = encode_want_datagram(content_id, args.chunk_ids)
        _emit_bytes(data, args.format, args.out)
        return 0

    if args.cmd == "parse-control":
        from holo.net.transport import CTRL_TYPE_INV, CTRL_TYPE_WANT, parse_control_datagram

        data = _read_input_bytes(hex_str=args.data_hex, b64_str=args.data_base64, path=args.input)
        parsed = parse_control_datagram(data)
        if parsed is None:
            print("invalid control datagram", file=sys.stderr)
            return 1
        ctrl_type, cid, chunk_ids = parsed
        ctrl_name = "inventory" if int(ctrl_type) == int(CTRL_TYPE_INV) else "want"
        print(json.dumps({"ctrl_type": ctrl_name, "content_id": cid.hex(), "chunk_ids": chunk_ids}))
        return 0

    if args.cmd == "assemble":
        from holo.net.transport import ChunkAssembler

        if args.out and args.out_dir:
            raise SystemExit("Use --out or --out-dir, not both.")
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_keys = {}
        if args.enc_key:
            enc_keys[int(args.enc_key_id)] = _parse_hex_bytes(args.enc_key)
        assembler = ChunkAssembler(auth_key=auth_key, enc_keys=enc_keys, max_partial_age=float(args.max_partial_age))
        completed = 0
        out_dir = Path(args.out_dir) if args.out_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        for datagram in _iter_encoded_lines(args.input, args.format):
            res = assembler.push_datagram(datagram)
            if res is None:
                continue
            cid, chunk_id, chunk_bytes = res
            completed += 1
            if out_dir:
                content_dir = out_dir / cid.hex()
                content_dir.mkdir(parents=True, exist_ok=True)
                out_path = content_dir / f"chunk_{int(chunk_id):04d}.holo"
                out_path.write_bytes(chunk_bytes)
                print(str(out_path))
            elif args.out:
                if completed > 1:
                    print("more than one chunk assembled; use --out-dir", file=sys.stderr)
                    return 1
                Path(args.out).write_bytes(chunk_bytes)
            else:
                print(f"{cid.hex()} {int(chunk_id)} {len(chunk_bytes)}")
        if completed == 0 and not args.out and not out_dir:
            print("no chunks completed", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "assembler-expire":
        from holo.net.transport import ChunkAssembler

        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_keys = {}
        if args.enc_key:
            enc_keys[int(args.enc_key_id)] = _parse_hex_bytes(args.enc_key)
        assembler = ChunkAssembler(auth_key=auth_key, enc_keys=enc_keys, max_partial_age=float(args.max_partial_age))
        for datagram in _iter_encoded_lines(args.input, args.format):
            assembler.push_datagram(datagram)
        now = time.time() + float(args.advance)
        assembler._expire_partials(now)
        print(json.dumps({"expired": assembler.counters["expired"], "partials": len(assembler._partials)}))
        return 0

    if args.cmd == "mesh-broadcast":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        if not args.peer:
            raise SystemExit("mesh-broadcast requires at least one --peer")
        peers = [_parse_addr(p) for p in args.peer]
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if args.bind:
            sock.bind(_parse_addr(args.bind))
        mesh = MeshNode(
            sock,
            CortexStore(args.store_root),
            peers=peers,
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            enc_key_id=int(args.enc_key_id),
        )
        mesh.broadcast_chunk_dir(
            args.uri,
            args.chunk_dir,
            repeats=int(args.repeats),
            priority=args.priority,
            send_recovery=bool(args.send_recovery),
        )
        print(f"sent chunks={mesh.counters['sent_chunks']} datagrams={mesh.counters['sent_datagrams']}")
        return 0

    if args.cmd == "mesh-recv":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        bind = _parse_addr(args.bind)
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        mcast_groups = [_parse_addr(m) for m in args.mcast]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(bind)
        sock.setblocking(False)
        mesh = MeshNode(
            sock,
            CortexStore(args.store_root),
            mcast_groups=mcast_groups,
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            enc_key_id=int(args.enc_key_id),
        )
        t0 = time.time()
        while time.time() - t0 < float(args.duration):
            res = mesh.recv_once()
            if res:
                cid, chunk_id, path = res
                print(f"stored chunk {int(chunk_id)} for cid {cid.hex()} at {path}")
            time.sleep(0.01)
        print(
            f"datagrams_seen={mesh._assembler.counters['datagrams']} stored_chunks={mesh.counters['stored_chunks']}"
        )
        return 0

    if args.cmd == "mesh-join-mcast":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        mcast_groups = [_parse_addr(m) for m in args.mcast]
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        MeshNode(sock, CortexStore("cortex_node"), mcast_groups=mcast_groups)
        t0 = time.time()
        while time.time() - t0 < float(args.duration):
            time.sleep(0.05)
        print(f"joined {len(mcast_groups)} multicast groups")
        return 0

    if args.cmd == "mesh-inventory":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        if not args.peer:
            raise SystemExit("mesh-inventory requires at least one --peer")
        peers = [_parse_addr(p) for p in args.peer]
        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if args.bind:
            sock.bind(_parse_addr(args.bind))
        mesh = MeshNode(
            sock,
            CortexStore(args.store_root),
            peers=peers,
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            enc_key_id=int(args.enc_key_id),
        )
        content_id = _content_id_from_args(args)
        mesh.send_inventory(content_id, peers=peers)
        print(f"sent inventory for {content_id.hex()} to {len(peers)} peers")
        return 0

    if args.cmd == "mesh-chunk-id":
        from holo.net.mesh import MeshNode

        print(int(MeshNode._chunk_id_from_name(args.path)))
        return 0

    if args.cmd == "mesh-recovery-id":
        from holo.net.mesh import MeshNode

        print(int(MeshNode._recovery_id_from_name(args.path)))
        return 0

    if args.cmd == "mesh-order-by-gain":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mesh = MeshNode(sock, CortexStore(args.store_root), max_payload=int(args.max_payload))
        chunk_dir = Path(args.chunk_dir)
        chunk_paths = [str(p) for p in sorted(chunk_dir.glob("chunk_*.holo"))]
        ordered = mesh._ordered_by_gain(str(chunk_dir), chunk_paths)
        for path in ordered:
            print(path)
        return 0

    if args.cmd == "mesh-local-chunk-ids":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mesh = MeshNode(sock, CortexStore(args.store_root), max_payload=int(args.max_payload))
        content_id = _content_id_from_args(args)
        ids = sorted(mesh._local_chunk_ids(content_id))
        print(" ".join(str(c) for c in ids))
        return 0

    if args.cmd == "mesh-handle-inventory":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mesh = MeshNode(
            sock,
            CortexStore(args.store_root),
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            enc_key_id=int(args.enc_key_id),
        )
        content_id = _content_id_from_args(args)
        peer = _parse_addr(args.peer)
        local = mesh._local_chunk_ids(content_id)
        missing = [c for c in args.chunk_ids if c not in local]
        mesh._handle_inventory(peer, content_id, args.chunk_ids)
        print(f"missing={len(missing)}")
        return 0

    if args.cmd == "mesh-handle-want":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        auth_key = _parse_hex_bytes(args.auth_key) if args.auth_key else None
        enc_key = _parse_hex_bytes(args.enc_key) if args.enc_key else None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mesh = MeshNode(
            sock,
            CortexStore(args.store_root),
            max_payload=int(args.max_payload),
            auth_key=auth_key,
            enc_key=enc_key,
            enc_key_id=int(args.enc_key_id),
        )
        content_id = _content_id_from_args(args)
        peer = _parse_addr(args.peer)
        mesh._handle_want(peer, content_id, args.chunk_ids)
        print(f"handled want for {len(args.chunk_ids)} chunks")
        return 0

    if args.cmd == "mesh-sendto":
        from holo.cortex.store import CortexStore
        from holo.net.mesh import MeshNode

        data = _read_input_bytes(hex_str=args.data_hex, b64_str=args.data_base64, path=args.input)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mesh = MeshNode(sock, CortexStore("cortex_node"))
        mesh._sendto(data, _parse_addr(args.peer))
        print(f"sent {len(data)} bytes to {args.peer}")
        return 0

    raise SystemExit(f"unknown net command: {args.cmd}")


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

    if argv and argv[0] == "net":
        return _run_net(argv[1:])

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
