import asyncio

from models import list_models

from experiments import export_experiment_data, metrics_predict_logic, save_movement, HandlerResult

from logger import logger

from render import render_image_raw, save_render_bytes

from models import get_model, ensure_started

from concurrent.futures import ThreadPoolExecutor

from encoding import encode_jpeg, encode_png

from statics import EXPERIMENTS_DIR

RENDER_EXECUTOR = ThreadPoolExecutor(max_workers=1)

from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse, Response, PlainTextResponse

async def models_page(request: Request):
    await ensure_started()
    return FileResponse("templates/models.html")

async def player_page(request: Request):
    await ensure_started()
    return FileResponse("templates/player.html")

async def get_list_of_all_available_models(request: Request):
    await ensure_started()
    models = await list_models()  
    return JSONResponse(models)

async def render_handler(request: Request):
    await ensure_started()

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
    profile = int(data.get("profile", 0))
    model = get_model(data.get("modelId"))

    loop = asyncio.get_running_loop()

    # 1) render raw in executor
    img_stream, render_ms = await loop.run_in_executor(
        RENDER_EXECUTOR,
        render_image_raw,
        azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height, profile, model
    )

    # 2) encode OUTSIDE render_image_raw
    factor = 1 << max(0, min(3, int(profile)))
    stream_quality = max(50, 70 - (factor * 5))

    jpeg_bytes = await loop.run_in_executor(
        RENDER_EXECUTOR,
        lambda: encode_jpeg(img_stream, quality=stream_quality)
    )

    return Response(
        jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Render-Time-Ms": f"{render_ms:.2f}"},
    )

    
def to_response(result: HandlerResult):
    headers = result.headers or {}

    if result.payload is not None:
        return JSONResponse(result.payload, status_code=result.status, headers=headers)

    if result.text is not None:
        return PlainTextResponse(result.text, status_code=result.status, headers=headers)

    return Response(b"", status_code=result.status, headers=headers)
    
    
# POST /metrics/predict
async def metrics_predict(request):
    await ensure_started()
    try:
        body = await request.json()
    except Exception as e:
        return PlainTextResponse(f"Invalid JSON: {e}", status_code=400)

    result = await metrics_predict_logic(body)
    
    return to_response(result)


# POST /metrics/predict
async def save_movements(request):
    await ensure_started()
    try:
        body = await request.json()
    except Exception as e:
        return PlainTextResponse(f"Invalid JSON: {e}", status_code=400)

    result = await save_movement(body)
    
    return to_response(result)


async def export_experiment(request : Request):
    await ensure_started()
    file_name = request.path_params["file_name"]

    result = await export_experiment_data(file_name)
    
    if result.status != 200 or result.content is None:
        return Response(b"", status_code=result.status)

    return Response(
        result.content,
        status_code=result.status,
        media_type=result.media_type,
        headers=result.headers or {},
    )


def model_to_json(model) -> dict: 
    # Return ONLY metadata / status. 
    # Do NOT return tensors. 
    return { 
        "id": model.id, 
        "name": getattr(model, "name", model.id), 
        "isLoaded": bool(getattr(model, "is_loaded", False)) or bool(getattr(model, "loaded", False))
    }

MODEL_LOAD_LOCK = asyncio.Lock()

async def load_model(request: Request):
    await ensure_started()
    data = await request.json()
    model_id = data.get("modelId")
    if not model_id:
        return JSONResponse({"error": "modelId missing"}, status_code=400)

    model = get_model(model_id=model_id)
    if not model:
        return JSONResponse({"error": f"unknown modelId={model_id}"}, status_code=404)

    async with MODEL_LOAD_LOCK:
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(None, model.load)

    return JSONResponse(model_to_json(model))


async def save_images(request: Request):
    await ensure_started()

    data = await request.json()

    experiment_name = data.get("experimentName")
    
    items = load_movements(experiment_name)
    
    for item in items:
        angle = item["angle"]
        elevation = item["elevation"] 
        x = item["x"]
        y = item["y"] 
        z = item["z"] 
        fx = item["fx"] 
        fy = item["fy"] 
        cx = item["cx"] 
        cy = item["cy"] 
        width = item["width"] 
        height = item["height"] 
        profile = item["profile"] 
        modelId = item["modelId"]
        model = get_model(modelId)

        loop = asyncio.get_running_loop()

        # 1) render raw in executor
        img_stream, render_ms = await loop.run_in_executor(
            RENDER_EXECUTOR,
            render_image_raw,
            angle, elevation, x, y, z, fx, fy, cx, cy, width, height, profile, model
        )

        # 2) encode OUTSIDE render_image_raw
        factor = 1 << max(0, min(3, int(profile)))
        stream_quality = max(50, 70 - (factor * 5))

        jpeg_bytes = await loop.run_in_executor(
            RENDER_EXECUTOR,
            lambda: encode_jpeg(img_stream, quality=stream_quality)
        )
        
        save_render_bytes(jpeg_bytes, str(modelId), base_name=experiment_name, type="jpg")

        png_bytes = await loop.run_in_executor(
            RENDER_EXECUTOR,
            lambda: encode_png(img_stream)
        )
        
        save_render_bytes(png_bytes, str(modelId), base_name=experiment_name, type= "png")

    return Response(
        headers={"Cache-Control": "no-store"},
    )
    
import json
from pathlib import Path
from typing import Any

def load_movements(path: str | Path) -> list[dict[str, Any]]:
    """
    Load movement items from an NDJSON file.

    Each line must be a valid JSON object.
    Returns a list of dicts with the parameters defined in the file.
    """
    items: list[dict[str, Any]] = []

    path = Path(f"{EXPERIMENTS_DIR}/{path}/movements.ndjson")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_no}"
                ) from e

    print(items)
    return items
