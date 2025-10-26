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

from starlette.applications import Starlette
from starlette.responses import Response, JSONResponse, PlainTextResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import FileResponse


ROOT = os.path.dirname(__file__)
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
                 fx, fy, cx, cy, width, height, profile) -> tuple[bytes, float]:
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 

    # Clamp profile and compute downscale factor (1, 2, 4, 8)
    p = max(0, min(3, int(profile)))
    factor = 1 << p

    w = int(width)
    h = int(height)

    print(f"[Render] Requested params -> azimuth={azimuth_deg}, elevation={elevation_deg}, "
          f"x={x}, y={y}, z={z}, fx={fx}, fy={fy}, cx={cx}, cy={cy}, "
          f"width={w}, height={h}, profile={profile} (factor={factor})")

    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).to(device).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    # --- timing start (sync first if on CUDA) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

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

    img = colors_rendered[0].detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)

    # Downsample AFTER rendering
    if factor > 1:
        out_w = max(1, w // factor)
        out_h = max(1, h // factor)
        pil_img = pil_img.resize((out_w, out_h), resample=Image.LANCZOS)
        print(f"[Render] Image size after downsampling: {pil_img.size}")
    else:
        print(f"[Render] No downsampling applied (profile={profile})")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    buf.seek(0)

    # --- timing end (sync again for accurate GPU timing) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    render_ms = (t1 - t0) * 1000.0

    print(f"[Render] Duration: {render_ms:.2f} ms")
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 

    return buf.getvalue(), render_ms

def save_render_bytes(jpeg_bytes: bytes, out_dir: str = "captures", base_name: str | None = None) -> str:
    """
    Save a JPEG byte buffer to disk and return the absolute file path.

    - out_dir is created if it doesn't exist (relative to this file by default).
    - base_name (without directories) can be provided; otherwise a timestamped name is used.
    """
    # Make out_dir absolute (under project root by default)
    out_dir_abs = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    if not base_name:
        # Unique, timestamped filename with millisecond precision
        ts_ms = int(time.time() * 1000)
        base_name = f"frame-{ts_ms}.jpg"

    # Ensure we only write under out_dir (no path traversal)
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

    print(f"[Handler] Received request data: {data}")

    # render in a worker thread and get bytes + duration
    jpeg_bytes, render_ms = await asyncio.to_thread(
        render_image, azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height, profile
    )
    
        # Save the image to disk (non-blocking for the event loop)
    # Example filename with size and profile; keep or simplify as you like.
    _saved_path = await asyncio.to_thread(
        save_render_bytes,
        jpeg_bytes,
        out_dir=os.path.join("static", "captures"),
        base_name=f"{int(time.time()*1000)}_{int(width)}x{int(height)}_p{profile}.jpg"
    )

    return Response(
        jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Render-Time-Ms": f"{render_ms:.2f}"
        },
    )


def get_current_kbps(dev: str = "wlp82s0") -> float | None:
    """
    Reads the current HTB class rate (in kbit/s) for the given network interface.

    Returns:
        float: current rate in kbit/s if found, otherwise None
    """
    try:
        out = subprocess.run(["tc", "class", "show", "dev", dev],
                             capture_output=True, text=True, check=False)
        text = out.stdout.strip() or out.stderr.strip()
        # Example line: "class htb 1:1 root rate 1200Kbit ceil 1200Kbit burst 15Kb ..."
        match = re.search(r"rate\s+([\d.]+)\s*Kbit", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"[tc error on {dev}]: {e}")
    return None


BASE_DIR = "experiments"

def now_ms(): return int(time.time() * 1000)

def ensure_exp_dir(exp_id: str):
    path = os.path.join(BASE_DIR, exp_id)
    os.makedirs(path, exist_ok=True)
    return path

# POST /metrics/predict
# body: { "expId": "exp-001", "pred_bps": 8500000, "profile": 2 }
async def metrics_predict(request):
    try:
        body = await request.json()
        exp_id    = str(body["expId"])
        pred_bps  = float(body["pred_bps"])
        profile   = body.get("profile")
        tc_status = get_current_kbps("wlp82s0")
    except Exception as e:
        return PlainTextResponse(f"Invalid JSON: {e}", status_code=400)

    d = ensure_exp_dir(exp_id)
    out_path = os.path.join(d, "pred.ndjson")

    rec = {
        "t_server": now_ms(),
        "expId": exp_id,
        "pred_bps": pred_bps,
        "profile": profile,
        "tc_status" : tc_status
    }
    with open(out_path, "a", buffering=1) as f:
        f.write(json.dumps(rec) + "\n")

    print(f"[predict] {rec}", flush=True)
    return JSONResponse({"ok": True})


starlette = Starlette(
    routes=[
        Route("/", home, methods=["GET"]),
        Route("/render", render_handler, methods=["POST"]),
        Route("/metrics/predict", metrics_predict, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
)

# the callable the aioquic server imports
async def app(scope, receive, send):
    await starlette(scope, receive, send)
