# ------------------- models functions -------------------


import time, threading, asyncio, torch
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from plyfile import PlyData
from typing import Optional

from logger import logger
from statics import MODELS_DIR, PREVIEW_CANDIDATES, EVICT_AFTER_MS, EVICT_CHECK_EVERY_S


def _now_ms() -> int:
    return int(time.time() * 1000)

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

@dataclass
class Model3D:
    # metadata (always present)
    id: str
    name: str
    preview_image_path: Optional[str]   # e.g. "/static/models/room/preview.jpg"
    model_path: str                     # absolute or relative path to .ply

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    in_use: int = 0

    # cache state
    loaded: bool = False
    loading: bool = False
    last_accessed: int = 0

    # tensors (only present when loaded)
    device: Optional[torch.device] = None
    means: Optional[torch.Tensor] = None        # (N, 3)
    quats: Optional[torch.Tensor] = None        # (N, 4)
    scales: Optional[torch.Tensor] = None       # (N, 3)
    opacities: Optional[torch.Tensor] = None    # (N,)
    shs: Optional[torch.Tensor] = None          # (N, 16, 3)
    
    
    def load(self, device: Optional[torch.device] = None) -> None:
        """Load tensors to device (defaults to cuda if available)."""
        if self.loaded:
            self.last_accessed = _now_ms()
            return

        means_np, quats_np, scales_np, opacities_np, shs_np = load_gs_ply(self.model_path)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.means = torch.from_numpy(means_np).to(device)
        self.quats = torch.from_numpy(quats_np).to(device)
        self.scales = torch.from_numpy(scales_np).to(device)
        self.opacities = torch.from_numpy(opacities_np).to(device).squeeze(-1)
        self.shs = torch.from_numpy(shs_np).to(device)
        self.device = device

        # free CPU numpy arrays
        del means_np, quats_np, scales_np, opacities_np, shs_np

        self.loaded = True
        self.last_accessed = _now_ms()
        
    def unload(self) -> bool:
        """Returns True if unloaded, False if skipped."""
        with self.lock:
            if not self.loaded:
                return False
            if self.in_use > 0:
                return False  # someone is rendering / using it

            self.loading = True
            try:
                self.means = None
                self.quats = None
                self.scales = None
                self.opacities = None
                self.shs = None
                self.device = None
                self.loaded = False
                self.last_accessed = _now_ms()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            finally:
                self.loading = False

        
        
    def get(self) -> tuple[torch.device, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return tensors, loading on demand."""
        if not self.loaded:
            self.load()
        assert self.device is not None
        assert self.means is not None
        assert self.quats is not None
        assert self.scales is not None
        assert self.opacities is not None
        assert self.shs is not None
        self.last_accessed = _now_ms()
        return (self.device, self.means, self.quats, self.scales, self.opacities, self.shs)
    
    def getLoading(self):
        return self.loading
    
    def acquire(self, device: Optional[torch.device] = None) -> tuple[torch.device, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Marks model as in use, ensures it is loaded, updates last_accessed,
        and returns tensors.
        """
        with self.lock:
            self.in_use += 1
            try:
                if not self.loaded:
                    self.load(device=device)
                self.last_accessed = _now_ms()
                assert self.device is not None
                assert self.means is not None
                assert self.quats is not None
                assert self.scales is not None
                assert self.opacities is not None
                assert self.shs is not None
                return (self.device, self.means, self.quats, self.scales, self.opacities, self.shs)
            except Exception:
                self.in_use -= 1
                raise

    def release(self) -> None:
        with self.lock:
            self.in_use = max(0, self.in_use - 1)
            self.last_accessed = _now_ms()


            
MODELS: dict[str, Model3D] = {}

def _first_existing_preview_file(model_dir: Path) -> str | None:
    for fn in PREVIEW_CANDIDATES:
        p = model_dir / fn
        if p.is_file():
            return p.name
        
    return None

def _first_ply_file(model_dir: Path) -> Optional[Path]:
    plys = sorted(model_dir.glob("*.ply"))
    return plys[0] if plys else None


def _first_ply_name_stem(model_dir: Path) -> Optional[str]:
    p = _first_ply_file(model_dir)
    return p.stem if p else None


def init_model_registry() -> dict[str, Model3D]:
    """
    Scan MODELS_DIR and create entries for ALL models, but do NOT load tensors.
    """
    global MODELS
    MODELS = {}

    base = Path(MODELS_DIR)
    if not base.is_dir():
        return MODELS

    for d in sorted(base.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue

        model_id = d.name
        ply_path = _first_ply_file(d)
        if ply_path is None:
            # skip folders without a .ply
            continue

        name = _first_ply_name_stem(d) or model_id
        preview_fn = _first_existing_preview_file(d)
        preview_url = f"/static/models/{model_id}/{preview_fn}" if preview_fn else None

        MODELS[model_id] = Model3D(
            id=model_id,
            name=name,
            preview_image_path=preview_url,
            model_path=str(ply_path),   # full path to .ply
            loaded=False,
            last_accessed=0,
        )
    logger.info("Models initialized!")


    return MODELS


def get_model(model_id: str) -> Model3D:
    if not MODELS:
        init_model_registry()
    return MODELS[model_id]


def load_model(model_id: str, device: Optional[torch.device] = None) -> None:
    if not MODELS:
        init_model_registry()
    MODELS[model_id].load(device=device)


def unload_model(model_id: str) -> None:
    if not MODELS:
        init_model_registry()
    MODELS[model_id].unload()


# ------------------- API helper -------------------

async def list_models() -> list[dict]:
    if not MODELS:
        init_model_registry()

    return [
        {
            "id": m.id,
            "name": m.name,
            "previewURL": m.preview_image_path,
        }
        for m in MODELS.values()
    ]

async def eviction_loop() -> None:
    while True:
        logger.info("Checking model usage.")
        now = _now_ms()
        for m in list(MODELS.values()):
            if m.loaded and (now - m.last_accessed) > EVICT_AFTER_MS:
                m.unload()
        await asyncio.sleep(EVICT_CHECK_EVERY_S)
        
        
_started = False
_started_lock = asyncio.Lock()

async def ensure_started():
    global _started
    if _started:
        return
    async with _started_lock:
        if _started:
            return
        init_model_registry()
        asyncio.create_task(eviction_loop())
        _started = True