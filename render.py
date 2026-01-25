import time, os, io

from gsplat import rasterization
from turbojpeg import TurboJPEG

import torch.nn.functional as F

import numpy as np

import torch
from scipy.spatial.transform import Rotation as R

from logger import logger
from statics import CAPTURES_DIR

jpeg = TurboJPEG()


def save_render_bytes(bytes: bytes, profile: int, base_name: str | None = None, type: str = 'jpg') -> str:
    os.makedirs(f"{CAPTURES_DIR}/{base_name}/{type}", exist_ok=True)
    ts_ms = int(time.time() * 1000)
    fileName = f"{base_name}/{type}/frame-{ts_ms}_{profile}_.{type}"

    out_path = os.path.join(CAPTURES_DIR, fileName)
    print(out_path)
    with open(out_path, "wb") as f:
        f.write(bytes)

    return out_path


def create_viewmat(azimuth_deg, elevation_deg, x, y, z):
    rot = R.from_euler("xyz", [elevation_deg, azimuth_deg, 0], degrees=True).as_matrix()
    trans = np.array([x, y, z])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = trans
    w2c = np.linalg.inv(c2w)
    return torch.tensor(w2c, dtype=torch.float32)



def render_image_raw(
    azimuth_deg, elevation_deg, x, y, z,
    fx, fy, cx, cy, width, height, profile, model
) -> tuple[np.ndarray, float]:
    """
    Returns:
      img_stream: np.uint8 HxWx3 RGB (CPU)
      render_ms: total render+downsample+transfer time in ms (no encode)
    """
    logger.debug("GPU memory before render: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    p = max(0, min(3, int(profile)))
    factor = 1 << p

    w = int(width)
    h = int(height)

    device, means, quats, scales, opacities, shs = model.acquire()

    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).to(device).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            colors_rendered, alphas, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=shs,
                viewmats=viewmat,
                Ks=K,
                width=w,
                height=h,
                packed=False,
                sh_degree=0,
                backgrounds=None,
                render_mode="RGB",
            )
    except Exception:
        logger.exception("Rasterization failed")
        raise
    finally:
        model.release()

    torch.cuda.synchronize()
    t_render = time.perf_counter()

    img_full_gpu_uint8 = (colors_rendered[0].clamp(0, 1) * 255).byte()

    # GPU downsample
    if factor > 1:
        low_h = max(1, h // factor)
        low_w = max(1, w // factor)

        img = img_full_gpu_uint8.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = F.interpolate(img, size=(low_h, low_w), mode="area")
        img_stream_gpu_uint8 = (img.squeeze(0).permute(1, 2, 0) * 255).byte()

        torch.cuda.synchronize()
        t_downsample = time.perf_counter()
    else:
        img_stream_gpu_uint8 = img_full_gpu_uint8
        t_downsample = t_render

    # Transfer stream image to CPU (raw RGB)
    img_stream = img_stream_gpu_uint8.cpu().numpy()  # np.uint8 HxWx3
    t_transfer = time.perf_counter()

    render_ms = (t_transfer - t0) * 1000.0

    logger.info(
        "[Render] total(no-encode)=%.2fms (raster=%.2fms, gpu_downsample=%.2fms, transfer=%.2fms)",
        render_ms,
        (t_render - t0) * 1000,
        (t_downsample - t_render) * 1000,
        (t_transfer - t_downsample) * 1000,
    )
    logger.debug("GPU memory after render: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    return img_stream, render_ms
