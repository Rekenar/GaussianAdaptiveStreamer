#!/usr/bin/env python3

'''
python compute_metrics.py \
  --test Metrics/"L2A Cascading Trace" \
  --ref Metrics/Original \
  --ffmpeg ffmpegF/ffmpeg \
  --out_csv results.csv
'''


import argparse, csv, json, math, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --------- helpers ---------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def list_images_sorted(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    files.sort(key=lambda p: natural_key(p.name))
    return files

def load_bgr(path: Path) -> np.ndarray:
    # cv2 imread handles many formats robustly
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img  # BGR uint8

def ensure_same_size(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    if ref.shape[:2] == test.shape[:2]:
        return test
    h, w = ref.shape[:2]
    return cv2.resize(test, (w, h), interpolation=cv2.INTER_AREA)

def compute_psnr_ssim(ref_bgr: np.ndarray, test_bgr: np.ndarray) -> Tuple[float, float]:
    """
    Compute PSNR and SSIM between two BGR images.
    PSNR is capped at 100 dB to avoid infinities when images are identical.
    """
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)

    # PSNR
    try:
        p = float(psnr(ref_rgb, test_rgb, data_range=255))
        if math.isinf(p) or p > 100:
            p = 100.0
    except Exception:
        p = float("nan")

    # SSIM
    try:
        s = float(ssim(ref_rgb, test_rgb, channel_axis=2, data_range=255))
    except TypeError:
        s = float(ssim(ref_rgb, test_rgb, multichannel=True, data_range=255))
    except Exception:
        s = float("nan")

    return p, s


def ffmpeg_has_libvmaf(ffmpeg_bin: str) -> bool:
    try:
        # If this fails, either ffmpeg is missing or libvmaf filter is missing
        subprocess.run(
            [ffmpeg_bin, "-v", "quiet", "-filters"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # cheaper probe: try getting help for libvmaf
        subprocess.run(
            [ffmpeg_bin, "-v", "quiet", "-h", "filter=libvmaf"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False

def save_png_rgb(arr_rgb: np.ndarray, path: Path):
    im = Image.fromarray(arr_rgb, mode="RGB")
    im.save(path, format="PNG")

def write_numbered_frames(images: List[Path], out_dir: Path, target_wh: Tuple[int,int]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    w, h = target_wh
    for idx, p in enumerate(images):
        bgr = load_bgr(p)
        if (bgr.shape[1], bgr.shape[0]) != (w, h):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        outp = out_dir / f"{idx:06d}.png"
        save_png_rgb(rgb, outp)
        written.append(outp)
    return written

def run_ffmpeg_vmaf_on_sequences(
    ffmpeg_bin: str,
    distorted_dir: Path,
    reference_dir: Path,
    frame_count: int,
    width: int,
    height: int,
    json_log: Path,
    model: Optional[str] = None,
    fps: int = 30
) -> List[float]:
    """
    Feed two image sequences (0:%06d.png) into libvmaf and return per-frame VMAF scores.
    The first input is distorted (test), the second is reference (original).
    """
    # explicit scaling in graph (safety)
    scale_dist = f"scale={width}:{height}:flags=bicubic"
    scale_ref  = f"scale={width}:{height}:flags=bicubic"

    # Build libvmaf options
    opts = [
        f"log_path={json_log.as_posix()}",
        "log_fmt=json",
        "n_threads=0",
    ]
    if model:
        mp = Path(model)
        if mp.exists():
            # A concrete .json model file was given
            opts.append(f"model_path={mp.as_posix()}")
        else:
            # Assume a named model version (e.g., vmaf_v0.6.1)
            # Newer ffmpeg/libvmaf accepts `model=version=...`
            opts.append(f"model=version={model}")

    # IMPORTANT: '=' after filter name
    vmaf_filter = f"libvmaf={':'.join(opts)}"
    lavfi = f"[0:v]{scale_dist}[d];[1:v]{scale_ref}[r];[d][r]{vmaf_filter}"

    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-r", str(fps), "-i", str((distorted_dir / "%06d.png").as_posix()),
        "-r", str(fps), "-i", str((reference_dir / "%06d.png").as_posix()),
        "-lavfi", lavfi,
        "-f", "null", "-"
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not json_log.exists():
        raise RuntimeError(f"ffmpeg/libvmaf failed:\nSTDERR:\n{proc.stderr}")

    data = json.loads(json_log.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    vmafs: List[float] = [float("nan")] * frame_count
    for fr in frames:
        idx = fr.get("frameNum")
        v = fr.get("metrics", {}).get("vmaf")
        if idx is not None and v is not None and 0 <= idx < frame_count:
            vmafs[idx] = float(v)
    return vmafs


# --------- main workflow ---------

def main():
    ap = argparse.ArgumentParser(
        description="Compute PSNR/SSIM per image pair and VMAF per frame by treating folders as sequences.")
    ap.add_argument("--ref", required=True, type=Path, help="Path to ORIGINAL images folder")
    ap.add_argument("--test", required=True, type=Path, help="Path to TEST images folder")
    ap.add_argument("--ffmpeg", required=True, type=str, help="Path to ffmpeg binary (must be built with libvmaf)")
    ap.add_argument("--out_csv", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--fps", type=int, default=30, help="FPS to feed sequences to VMAF (default: 30)")
    ap.add_argument("--model", type=str, default=None,
                    help="Optional libvmaf model spec (e.g., 'model=vmaf_v0.6.1' or absolute path to .json).")
    args = ap.parse_args()

    ref_dir: Path = args.ref
    test_dir: Path = args.test
    ffmpeg_bin: str = args.ffmpeg
    out_csv: Path = args.out_csv
    fps: int = args.fps
    model: Optional[str] = args.model

    if not ref_dir.is_dir():
        print(f"[error] --ref folder not found: {ref_dir}", file=sys.stderr); sys.exit(2)
    if not test_dir.is_dir():
        print(f"[error] --test folder not found: {test_dir}", file=sys.stderr); sys.exit(2)

    ref_imgs = list_images_sorted(ref_dir)
    test_imgs = list_images_sorted(test_dir)
    if not ref_imgs or not test_imgs:
        print("[error] No images found in one or both folders.", file=sys.stderr); sys.exit(2)

    n = min(len(ref_imgs), len(test_imgs))
    if len(ref_imgs) != len(test_imgs):
        print(f"[warn] Different counts: ref={len(ref_imgs)} test={len(test_imgs)}. "
              f"Proceeding with first {n} pairs.", file=sys.stderr)

    pairs = list(zip(ref_imgs[:n], test_imgs[:n]))

    # PSNR/SSIM per pair; also collect unified dimensions for VMAF
    per_filename = []
    psnr_vals, ssim_vals = [], []

    # Reference resolution = first ref img
    first_ref = load_bgr(pairs[0][0])
    H, W = first_ref.shape[:2]

    for ref_p, test_p in pairs:
        ref_bgr = load_bgr(ref_p)
        if ref_bgr.shape[:2] != (H, W):
            # Keep VMAF consistent: stick to the FIRST ref size
            ref_bgr = cv2.resize(ref_bgr, (W, H), interpolation=cv2.INTER_AREA)

        test_bgr = load_bgr(test_p)
        test_bgr = ensure_same_size(ref_bgr, test_bgr)

        pval, sval = compute_psnr_ssim(ref_bgr, test_bgr)

        per_filename.append(ref_p.name)  # name from ref side; adjust if you prefer test
        psnr_vals.append(pval)
        ssim_vals.append(sval)

    # VMAF via ffmpeg+libvmaf
    have_vmaf = ffmpeg_has_libvmaf(ffmpeg_bin)
    vmaf_vals = [float("nan")] * n

    if not have_vmaf:
        print("[warn] ffmpeg missing libvmaf filter or not found. VMAF will be 'nan'.", file=sys.stderr)
    else:
        with tempfile.TemporaryDirectory(prefix="vmaf_seq_") as tmpdir:
            tmp = Path(tmpdir)
            ref_frames = tmp / "ref_frames"
            test_frames = tmp / "test_frames"

            # Write numbered, size-aligned PNG frames
            write_numbered_frames([p for p, _ in pairs], ref_frames, (W, H))
            write_numbered_frames([p for _, p in pairs], test_frames, (W, H))

            json_log = tmp / "vmaf_log.json"
            try:
                vmaf_vals = run_ffmpeg_vmaf_on_sequences(
                    ffmpeg_bin=ffmpeg_bin,
                    distorted_dir=test_frames,
                    reference_dir=ref_frames,
                    frame_count=n,
                    width=W, height=H,
                    json_log=json_log,
                    model=model,
                    fps=fps
                )
            except Exception as e:
                print(f"[warn] VMAF computation failed: {e}\n"
                      f"VMAF column will be 'nan'.", file=sys.stderr)
                vmaf_vals = [float("nan")] * n

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "psnr", "ssim", "vmaf"])
        for name, pval, sval, vval in zip(per_filename, psnr_vals, ssim_vals, vmaf_vals):
            w.writerow([name, f"{pval:.6f}" if not math.isnan(pval) else "nan",
                              f"{sval:.6f}" if not math.isnan(sval) else "nan",
                              f"{vval:.6f}" if not math.isnan(vval) else "nan"])

    # Print summary
    def safe_avg(vals):
        arr = [v for v in vals if not math.isnan(v)]
        return (sum(arr) / len(arr)) if arr else float("nan")

    avg_psnr = safe_avg(psnr_vals)
    avg_ssim = safe_avg(ssim_vals)
    avg_vmaf = safe_avg(vmaf_vals)

    print(f"Wrote per-image metrics to: {out_csv}")
    print(f"Pairs compared: {n}")
    print(f"Averages: PSNR={avg_psnr:.3f}  SSIM={avg_ssim:.5f}  VMAF={avg_vmaf:.2f}")

if __name__ == "__main__":
    main()
