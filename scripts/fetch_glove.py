#!/usr/bin/env python3
"""Download GloVe 6B 100d into a vectors directory (large; not committed)."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

SOURCE = "https://nlp.stanford.edu/data/glove.6B.zip"
# zip sha from fly-cast PROTOCOL; extracted 100d line file checked loosely by size
ZIP_SHA256 = "95dde4dfd627ab26608d33e76d1195ec059734bd29089ea52cadb08d07c64544"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="Vectors directory")
    args = ap.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    zpath = args.dir / "glove.6B.zip"
    out = args.dir / "glove.6B.100d.txt"
    if out.is_file() and out.stat().st_size > 100_000_000:
        print(f"already have {out}")
        return 0
    if not zpath.is_file():
        print(f"downloading {SOURCE} …")
        urlretrieve(SOURCE, zpath)
    got = sha256_file(zpath)
    if got != ZIP_SHA256:
        raise SystemExit(f"zip sha256 mismatch: {got}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extract("glove.6B.100d.txt", path=args.dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
