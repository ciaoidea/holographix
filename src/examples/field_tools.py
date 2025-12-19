"""
Utility helpers to inspect and curate holographic fields (.holo directories).

Subcommands:
  list   - print chunk IDs present
  drop   - delete specific chunk IDs
  copy   - copy a subset of chunks to another directory
  merge  - merge chunks from multiple dirs into one
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def list_chunks(holo_dir: Path) -> List[int]:
    return sorted(int(p.stem.split("_")[1]) for p in holo_dir.glob("chunk_*.holo"))


def cmd_list(args: argparse.Namespace) -> None:
    holo_dir = Path(args.dir)
    if not holo_dir.exists():
        raise FileNotFoundError(holo_dir)
    chunks = list_chunks(holo_dir)
    print(f"{holo_dir}: {len(chunks)} chunks")
    print(" ".join(str(c) for c in chunks))


def cmd_drop(args: argparse.Namespace) -> None:
    holo_dir = Path(args.dir)
    ids = set(int(x) for x in args.chunk_ids)
    count = 0
    for cid in ids:
        path = holo_dir / f"chunk_{cid:04d}.holo"
        if path.exists():
            path.unlink()
            count += 1
    print(f"Deleted {count} chunks from {holo_dir}")


def cmd_copy(args: argparse.Namespace) -> None:
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    ids = set(int(x) for x in args.chunk_ids) if args.chunk_ids else None
    copied = 0
    for p in src.glob("chunk_*.holo"):
        cid = int(p.stem.split("_")[1])
        if ids is not None and cid not in ids:
            continue
        shutil.copyfile(p, dst / p.name)
        copied += 1
    print(f"Copied {copied} chunks to {dst}")


def cmd_merge(args: argparse.Namespace) -> None:
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for directory in args.src:
        src = Path(directory)
        for p in src.glob("chunk_*.holo"):
            target = dst / p.name
            if target.exists():
                continue
            shutil.copyfile(p, target)
            copied += 1
    print(f"Merged {copied} chunks into {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect and curate holographic .holo directories.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List chunk IDs in a .holo directory.")
    p_list.add_argument("dir")
    p_list.set_defaults(func=cmd_list)

    p_drop = sub.add_parser("drop", help="Delete specific chunk IDs from a .holo directory.")
    p_drop.add_argument("dir")
    p_drop.add_argument("chunk_ids", nargs="+")
    p_drop.set_defaults(func=cmd_drop)

    p_copy = sub.add_parser("copy", help="Copy selected chunks to another directory.")
    p_copy.add_argument("src")
    p_copy.add_argument("dst")
    p_copy.add_argument("--chunk-ids", nargs="*", help="Specific chunk IDs to copy (default: all).")
    p_copy.set_defaults(func=cmd_copy)

    p_merge = sub.add_parser("merge", help="Merge chunks from multiple dirs into one destination.")
    p_merge.add_argument("dst")
    p_merge.add_argument("src", nargs="+")
    p_merge.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
