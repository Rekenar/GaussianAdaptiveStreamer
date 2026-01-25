import time, os, io

from gsplat import rasterization
from PIL import Image
from turbojpeg import TurboJPEG

import numpy as np

import torch
from scipy.spatial.transform import Rotation as R

from models import Model3D

from logger import logger
from statics import CAPTURES_DIR


def save_render_bytes(jpeg_bytes: bytes, base_name: str | None = None, type: str = '.jpg') -> str:
    if not base_name:
        ts_ms = int(time.time() * 1000)
        base_name = f"frame-{ts_ms}{type}"

    out_path = os.path.join(CAPTURES_DIR, base_name)

    with open(out_path, "wb") as f:
        f.write(jpeg_bytes)

    return out_path


def create_viewmat(azimuth_deg, elevation_deg, x, y, z):
    rot = R.from_euler("xyz", [elevation_deg, azimuth_deg, 0], degrees=True).as_matrix()
    trans = np.array([x, y, z])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = trans
    w2c = np.linalg.inv(c2w)
    return torch.tensor(w2c, dtype=torch.float32)

def render_image(azimuth_deg, elevation_deg, x, y, z,
                 fx, fy, cx, cy, width, height, profile, savePNG, saveJPG, model: Model3D) -> tuple[bytes, float, bytes, bytes]:
    
    logger.debug("GPU memory before render: %.2f GB", torch.cuda.memory_allocated() / 1024**3)
        
    p = max(0, min(3, int(profile)))
    factor = 1 << p
    
    w = int(width)
    h = int(height)
    
    logger.info(
        "[Render] params az=%.1f el=%.1f pos=(%.2f,%.2f,%.2f) "
        "size=%dx%d profile=%d factor=%d",
        azimuth_deg, elevation_deg, x, y, z,
        w, h, profile, factor
    )
    
    device, means, quats, scales, opacities, shs = model.acquire()
    
    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).to(device).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    logger.info("Calling rasterization at full resolution %dx%d...", w, h)
    
    with torch.no_grad():
        try:
            colors_rendered, alphas, _ = rasterization(
                means = means,
                quats = quats,
                scales = scales,
                opacities = opacities,
                colors = shs,
                viewmats = viewmat,
                Ks = K,
                width = w,
                height = h,
                packed = False,
                sh_degree = 0,
                backgrounds = None,
                render_mode = "RGB",
            )
        except Exception:
            logger.exception("Rasterization failed")
            raise
        finally:
            model.release()
    
    torch.cuda.synchronize()
    t_render = time.perf_counter()
    
    img_full_gpu_uint8 = (colors_rendered[0].clamp(0, 1) * 255).byte()
    
    # GPU-based downsampling for streaming
    if factor > 1:
        img_downsampled = img_full_gpu_uint8.permute(2, 0, 1).unsqueeze(0) / 255.0
        
        low_h = max(1, h // factor)
        low_w = max(1, w // factor)
        
        img_downsampled = torch.nn.functional.interpolate(
            img_downsampled,
            size=(low_h, low_w),
            mode='area', 
            align_corners=None
        )
        
        img_downsampled = (img_downsampled.squeeze(0).permute(1, 2, 0) * 255).byte()
        
        t_downsample = time.perf_counter()
                
        img_stream = img_downsampled.cpu().numpy()
        
        logger.info("[Render] downsampled to %dx%d on GPU", low_w, low_h)
    else:
        t_downsample = t_render
        img_stream = img_full_gpu_uint8.cpu().numpy()

    
    t_transfer_stream = time.perf_counter()
    
    pil_img_stream = Image.fromarray(img_stream)
    
    buf_jpg_original = io.BytesIO()
    buf_png_original = io.BytesIO()
    
    # For saved files, use full resolution
    if saveJPG or savePNG:
        img_full = img_full_gpu_uint8.cpu().numpy()
        pil_img_full = Image.fromarray(img_full)
        t_transfer_full = time.perf_counter()
        
        jpeg_encoder = TurboJPEG()

        
        if savePNG:
            pil_img_full.save(buf_png_original, format="PNG", optimize=True)
            buf_png_original.seek(0)
        
        if saveJPG:
            buf_jpg_original = io.BytesIO(jpeg_encoder.encode(img_full, quality=95))

    else:
        t_transfer_full = t_transfer_stream
    
    # Stream response uses downsampled image
    stream_quality = max(50, 70 - (factor * 5))  # 70, 65, 60, 55 for factors 1,2,4,8

    buf_jpg = io.BytesIO()
    pil_img_stream.save(buf_jpg, format="JPEG", quality=stream_quality, optimize=False)
    buf_jpg.seek(0)
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    render_ms = (t1 - t0) * 1000.0
    
    logger.info(
        "[Render] total=%.2fms (raster=%.2fms, gpu_downsample=%.2fms, "
        "transfer_stream=%.2fms, transfer_full=%.2fms, encode=%.2fms)",
        render_ms,
        (t_render - t0) * 1000,
        (t_downsample - t_render) * 1000,
        (t_transfer_stream - t_downsample) * 1000,
        (t_transfer_full - t_transfer_stream) * 1000,
        (t1 - t_transfer_full) * 1000
    )
    
    logger.debug("GPU memory after render: %.2f GB", torch.cuda.memory_allocated() / 1024**3)
    
    return buf_jpg.getvalue(), render_ms, buf_jpg_original.getvalue(), buf_png_original.getvalue()




