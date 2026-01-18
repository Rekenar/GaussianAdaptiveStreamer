# gs_app.py

'''
python http3_server.py --certificate certificates/ssl_cert.pem --private-key certificates/ssl_key.pem --host 0.0.0.0
'''
'''
 google-chrome \
  --enable-experimental-web-platform-features \
  --ignore-certificate-errors-spki-list=BSQJ0jkQ7wwhR7KvPZ+DSNk2XTZ/MS6xCbo9qu++VdQ= \
  --origin-to-force-quic-on=localhost:4433 \
  https://localhost:4433/
'''


import io, os, asyncio, json, time, subprocess, re
import numpy as np
from plyfile import PlyData
import torch
from scipy.spatial.transform import Rotation as R
from gsplat import rasterization
from PIL import Image
import zipfile


from concurrent.futures import ThreadPoolExecutor

RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1)

from starlette.applications import Starlette
from starlette.responses import Response, JSONResponse, PlainTextResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import FileResponse

import logging
from logging.handlers import RotatingFileHandler

from datetime import datetime


ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)



today = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_DIR, f"gs_server_{today}.log")

logging.basicConfig(level=logging.INFO)  # root config

logger = logging.getLogger("gs")
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=50 * 1024 * 1024,  # 50 MB
    backupCount=5,
    encoding="utf-8",
)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


logger.warning("CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES"))
logger.warning("CUDA_LAUNCH_BLOCKING=%s", os.getenv("CUDA_LAUNCH_BLOCKING"))
logger.warning("TORCH_SHOW_CPP_STACKTRACES=%s", os.getenv("TORCH_SHOW_CPP_STACKTRACES"))
logger.warning("torch=%s cuda=%s devices=%d device0=%s",
                torch.__version__, torch.cuda.is_available(),
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")


TEMPLATES_DIR = os.path.join(ROOT, "templates")
STATIC_DIR = os.path.join(ROOT, "static")

# -------------------- helpers --------------------
def load_gs_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    means = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=-1).astype(np.float32)
    scales = np.exp(np.stack((vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]), axis=-1)).astype(np.float32)
    quats = np.stack((vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]), axis=-1).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    opacities = 1 / (1 + np.exp(-vertex["opacity"])).astype(np.float32)
    shs = np.zeros((means.shape[0], 16, 3), dtype=np.float32)
    shs[:, 0, 0] = vertex["f_dc_0"]
    shs[:, 0, 1] = vertex["f_dc_1"]
    shs[:, 0, 2] = vertex["f_dc_2"]
    for i in range(45):
        shs[:, (i // 3) + 1, i % 3] = vertex[f"f_rest_{i}"]
    return means, quats, scales, opacities, shs

def create_viewmat(azimuth_deg, elevation_deg, x, y, z):
    rot = R.from_euler("xyz", [elevation_deg, azimuth_deg, 0], degrees=True).as_matrix()
    trans = np.array([x, y, z])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = trans
    w2c = np.linalg.inv(c2w)
    return torch.tensor(w2c, dtype=torch.float32)

def chk(name, t):
    if not torch.is_tensor(t):
        logger.info("%s: (not tensor) %r", name, type(t))
        return
    finite = bool(torch.isfinite(t).all())
    logger.info("%s: shape=%s dtype=%s device=%s contig=%s finite=%s min=%s max=%s",
                 name, tuple(t.shape), t.dtype, t.device, t.is_contiguous(), finite,
                 t.min().item() if t.numel() else "n/a",
                 t.max().item() if t.numel() else "n/a")
    if not finite:
        raise ValueError(f"{name} has NaN/Inf")

# ---------- load model once on import ----------
MODEL_PLY = os.path.join(STATIC_DIR, "models", "model_high.ply")
means_np, quats_np, scales_np, opacities_np, shs_np = load_gs_ply(MODEL_PLY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

means = torch.from_numpy(means_np).to(device)
quats = torch.from_numpy(quats_np).to(device)
scales = torch.from_numpy(scales_np).to(device)
opacities = torch.from_numpy(opacities_np).to(device).squeeze(-1)
shs = torch.from_numpy(shs_np).to(device)

# free host copies
del means_np, quats_np, scales_np, opacities_np, shs_np

def render_image(azimuth_deg, elevation_deg, x, y, z,
                 fx, fy, cx, cy, width, height, profile, savePNG, saveJPG) -> tuple[bytes, float, bytes, bytes]:
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True) 

    logger.debug(
        "GPU memory before render: %.2f GB",
        torch.cuda.memory_allocated() / 1024**3
    )

    # Clamp profile and compute downscale factor (1, 2, 4, 8)
    p = max(0, min(3, int(profile)))
    factor = 1 << p

    w = int(width)
    h = int(height)

    logger.info(
    "[Render] params az=%.1f el=%.1f pos=(%.2f,%.2f,%.2f) "
    "fx=%.1f fy=%.1f cx=%.1f cy=%.1f size=%dx%d profile=%d factor=%d",
    azimuth_deg, elevation_deg, x, y, z,
    fx, fy, cx, cy, w, h, profile, factor
    )

    print(f"[Render] Requested params -> azimuth={azimuth_deg}, elevation={elevation_deg}, "
          f"x={x}, y={y}, z={z}, fx={fx}, fy={fy}, cx={cx}, cy={cy}, "
          f"width={w}, height={h}, profile={profile} (factor={factor})", flush=True)

    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).to(device).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    logger.info("Before rasterization")
    
    #chk("means", means)
    #chk("scales", scales)
    #chk("quats", quats)
    #chk("opacities", opacities)
    #chk("colors", shs)
    #chk("viewmat", viewmat)
    #chk("viewmat", viewmat)


    
    logger.info("Calling rasterization now...")
    try:
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
        logger.info("Returned from rasterization, syncing...")
        torch.cuda.synchronize()
        logger.info("After synchronize")
    except Exception:
        logger.exception("Rasterization failed")
        raise

    
    logger.info("After rasterization")

    img = colors_rendered[0].detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    
    # Downsample AFTER rendering
    if factor > 1:
        low_w = max(1, w // factor)
        low_h = max(1, h // factor)

        # Downsample (simulate lower render quality)
        pil_img = pil_img.resize((low_w, low_h), resample=Image.LANCZOS)
        print(f"[Render] Image size after downsampling: {pil_img.size}", flush=True)
        logger.debug(
            "[Render] downsampled image size: %s",
            pil_img.size
        )
        if saveJPG or savePNG:
            # Upscale back to original resolution for images
            pil_img = pil_img.resize((w, h), resample=Image.LANCZOS)

    else:
        print(f"[Render] No downsampling applied (profile={profile})", flush=True)
        logger.debug(
            "[Render] no downsampling applied (profile=%d)",
            profile
        )

    buf_jpg_original = io.BytesIO()

    buf_png_original = io.BytesIO()

    if savePNG: 
        pil_img.save(buf_png_original, format="PNG")
        buf_png_original.seek(0)
        
    if saveJPG:
        pil_img.save(buf_jpg_original, format="JPEG")
        buf_jpg_original.seek(0)
    
    buf_jpg = io.BytesIO()
    pil_img.save(buf_jpg, format="JPEG", quality=70, subsampling=0)
    buf_jpg.seek(0)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    render_ms = (t1 - t0) * 1000.0

    print(f"[Render] Duration: {render_ms:.2f} ms", flush=True)
    logger.info(
        "[Render] duration: %.2f ms",
        render_ms
    )   
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True) 
    
    logger.debug(
        "GPU memory after render: %.2f GB",
        torch.cuda.memory_allocated() / 1024**3
    )

    
    return buf_jpg.getvalue(), render_ms, buf_jpg_original.getvalue(), buf_png_original.getvalue()


def save_render_bytes(jpeg_bytes: bytes, out_dir: str = "captures", base_name: str | None = None, type: str = '.jpg') -> str:
    out_dir_abs = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    if not base_name:
        ts_ms = int(time.time() * 1000)
        base_name = f"frame-{ts_ms}{type}"

    base_name = os.path.basename(base_name)
    out_path = os.path.join(out_dir_abs, base_name)

    with open(out_path, "wb") as f:
        f.write(jpeg_bytes)

    return out_path


# ------------------- HTTP handlers -------------------
async def home(request:Request):
    return FileResponse("templates/index.html")

async def render_handler(request: Request):
    data = await request.json()
    
    print(f"[Handler] Received request data: {data}", flush=True)
    
    logger.info(
        "[Handler] render request received: %s",
        data
    )
        
    azimuth = float(data.get("angle", 180))
    elevation = float(data.get("elevation", 0))
    x = float(data.get("x", 0))
    y = float(data.get("y", 0))
    z = float(data.get("z", 5.0))
    fx = float(data.get("fx", 1300.0))
    fy = float(data.get("fy", 800.0))
    cx = float(data.get("cx", 400.0))
    cy = float(data.get("cy", 300.0))
    width = float(data.get("width", 800))
    height = float(data.get("height", 600))
    profile = int(data.get("profile", 0))  # 0..3 -> 1x,2x,4x,8x downsample
    fileName = data.get("fileName") or "default"
    savePNG = bool(data.get("savePNG", False))
    saveJPG = bool(data.get("saveJPG", False))

    loop = asyncio.get_running_loop()
    jpeg_bytes, render_ms, jpeg_original, png_original = await loop.run_in_executor(
        RENDER_EXECUTOR,
        render_image,
        azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height, profile, savePNG, saveJPG
    )
    

    if savePNG:
        logger.info("Saving PNG")
        save_render_bytes(png_original, f"captures/{fileName}/png", type=".png")

    if saveJPG:
        logger.info("Saving JPGE")
        save_render_bytes(jpeg_bytes=jpeg_original, out_dir=f"captures/{fileName}/jpg", type=".jpg")

    return Response(
        jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Render-Time-Ms": f"{render_ms:.2f}"},
    )


def get_current_kbps(dev: str = "wlp82s0") -> float | None:
    try:
        out = subprocess.run(["tc", "class", "show", "dev", dev],
                             capture_output=True, text=True, check=False)
        text = out.stdout.strip() or out.stderr.strip()
        match = re.search(r"rate\s+([\d.]+)\s*Kbit", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"[tc error on {dev}]: {e}", flush=True)
        logger.warning(
            "[tc] failed to read tc status on dev=%s: %s",
            dev, e
        )

    return None


def now_ms(): 
    return int(time.time() * 1000)

EXPERIMENT_DIR = os.path.join(ROOT, "experiment")


# POST /metrics/predict
# body: { "expId": "exp-001", "pred_bps": 8500000, "profile": 2 }
async def metrics_predict(request):
    try:
        body = await request.json()
        pred_bps  = float(body["pred_bps"])
        profile   = body.get("profile")
        file_name = body.get("fileName") or "default"
        network = body.get("networkName")
        render_ms = body.get("renderMs")
        tc_status = get_current_kbps(network)
    except Exception as e:
        return PlainTextResponse(f"Invalid JSON: {e}", status_code=400)

    logger.info(f"Request with data: {body}")

    out_path = os.path.join(EXPERIMENT_DIR, f"{file_name}")
    os.makedirs(out_path, exist_ok=True)

    rec = {
        "t_server": now_ms(),
        "pred_bps": pred_bps,
        "profile": profile,
        "tc_status" : tc_status,
        "renderMs" : render_ms
    }
    with open(f"{out_path}/testdata.ndjson", "a", buffering=1) as f:
        f.write(json.dumps(rec) + "\n")

    print(f"[predict] file={file_name} {rec}", flush=True)
    logger.info(
        "[predict] file=%s record=%s",
        file_name, rec
    )

    return JSONResponse({"ok": True, "file": file_name})

async def export_experiment(request: Request):
    file_name = request.query_params.get("fileName")
    if not file_name:
        return Response(b"", status_code=404)

    cap_dir = os.path.join(ROOT, "captures", file_name)
    ndjson_path = os.path.join(EXPERIMENT_DIR, file_name, "testdata.ndjson")

    if not os.path.isdir(cap_dir) or not os.path.isfile(ndjson_path):
        return Response(b"", status_code=404)

    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(ndjson_path, arcname=os.path.join("experiment", file_name, "testdata.ndjson"))

        for root_dir, _, files in os.walk(cap_dir):
            for fn in files:
                full_path = os.path.join(root_dir, fn)
                rel_path = os.path.relpath(full_path, ROOT)
                z.write(full_path, arcname=rel_path)

    zip_bytes = buf.getvalue()

    return Response(
        zip_bytes,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="{file_name}.zip"',
        },
    )



starlette = Starlette(
    routes=[
        Route("/", home, methods=["GET"]),
        Route("/render", render_handler, methods=["POST"]),
        Route("/metrics/predict", metrics_predict, methods=["POST"]),
        Route("/export", export_experiment, methods=["GET"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
        
    ]
)

# the callable the aioquic server imports
async def app(scope, receive, send):
    await starlette(scope, receive, send)
