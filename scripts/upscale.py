#!/usr/bin/env python3
"""
Upscale p1/p2/p3 images IN-PLACE to p0 size using Lanczos.

- Naming must end with _p0/_p1/_p2/_p3 (or -pX or .pX) before the extension.
- Ladder: p0=1x, p1=2x, p2=4x, p3=8x
- Overwrites the original file (same name).

Usage:
  python upscale.py /path/to/folder [--recursive] [--dry-run]
"""

import argparse, re, tempfile, shutil
from pathlib import Path
from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
TAG_RE = re.compile(r"(?:^|[_\-.])p([0-3])$")  # matches trailing ..._pX (before extension)

def detect_px_tag(stem: str):
    m = TAG_RE.search(stem)
    if not m:
        return None
    return int(m.group(1))  # 0..3

def upscale_file_inplace(src: Path, level: int, dry_run: bool=False) -> bool:
    factor = 2 ** level
    if factor == 1:
        print(f"[skip] {src.name}: already p0")
        return False

    if dry_run:
        print(f"[dry-run] {src.name}: upscale x{factor} (Lanczos) -> overwrite")
        return True

    # Write to a temp file first, then atomically replace
    tmp = None
    try:
        with Image.open(src) as im:
            w, h = im.size
            new_size = (w * factor, h * factor)

            up = im.resize(new_size, Image.Resampling.LANCZOS)

            exif = im.info.get("exif")
            save_kwargs = {}
            if exif:
                save_kwargs["exif"] = exif
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs.setdefault("quality", 95)
                save_kwargs.setdefault("subsampling", 0)
                save_kwargs.setdefault("progressive", True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=src.suffix, dir=str(src.parent)) as tf:
                tmp = Path(tf.name)
            up.save(tmp, **save_kwargs)
            shutil.move(str(tmp), str(src))

            print(f"[ok] {src.name}: {w}x{h} → {new_size[0]}x{new_size[1]} (in-place)")
            return True
    except Exception as e:
        print(f"[error] {src}: {e}")
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)
        return False

def main():
    ap = argparse.ArgumentParser(description="Upscale p1/p2/p3 images in-place to p0 using Lanczos.")
    ap.add_argument("folder", type=Path, help="Folder containing images")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    args = ap.parse_args()

    if not args.folder.is_dir():
        ap.error(f"Not a directory: {args.folder}")

    it = args.folder.rglob("*") if args.recursive else args.folder.glob("*")

    considered = 0
    changed = 0
    for p in it:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        level = detect_px_tag(p.stem)
        if level is None:
            continue  # not tagged; skip to avoid wrong scaling
        considered += 1
        if upscale_file_inplace(p, level, args.dry_run):
            changed += 1

    print(f"\nDone. Considered: {considered}, upscaled (overwritten): {changed}, skipped: {considered - changed}")

if __name__ == "__main__":
    main()
