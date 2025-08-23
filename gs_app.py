# gs_app.py
# python http3_server.py --certificate certificates/ssl_cert.pem --private-key certificates/ssl_key.pem
import io, os, asyncio
import numpy as np
from plyfile import PlyData
import torch
from scipy.spatial.transform import Rotation as R
from gsplat import rasterization
from PIL import Image

from starlette.applications import Starlette
from starlette.responses import Response
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

def render_image(azimuth_deg, elevation_deg, x, y, z, fx, fy, cx, cy, width, height) -> bytes:
    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).to(device).unsqueeze(0)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    colors_rendered, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=shs,
        viewmats=viewmat,
        Ks=K,
        width=int(width),
        height=int(height),
        packed=False,
        sh_degree=0,
        backgrounds=None,
        render_mode="RGB",
    )
    img = colors_rendered[0].detach().cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    return buf.getvalue()

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

    # Offload to a worker thread so the event loop stays responsive
    jpeg_bytes = await asyncio.to_thread(
        render_image, azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height
    )
    return Response(
        jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

starlette = Starlette(
    routes=[
        Route("/", home, methods=["GET"]),
        Route("/render", render_handler, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
)

# the callable the aioquic server imports
async def app(scope, receive, send):
    await starlette(scope, receive, send)
