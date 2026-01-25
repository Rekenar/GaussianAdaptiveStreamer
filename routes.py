import asyncio

from models import list_models

from experiments import export_experiment_data, metrics_predict_logic

from logger import logger

from render import render_image

from models import get_model, ensure_started

from concurrent.futures import ThreadPoolExecutor

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
    model = get_model(data.get("modelId"))
    
    loop = asyncio.get_running_loop()
    jpeg_bytes, render_ms, jpeg_original, png_original = await loop.run_in_executor(
        RENDER_EXECUTOR,
        render_image,
        azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height, profile, savePNG, saveJPG, model
    )

    if savePNG:
        logger.info("Saving PNG")
        #save_render_bytes(png_original, f"captures/{fileName}/png", type=".png")

    if saveJPG:
        logger.info("Saving JPGE")
        #save_render_bytes(jpeg_bytes=jpeg_original, out_dir=f"captures/{fileName}/jpg", type=".jpg")

    return Response(
        jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store", "X-Render-Time-Ms": f"{render_ms:.2f}"},
    )
    
    
# POST /metrics/predict
async def metrics_predict(request):
    await ensure_started()
    try:
        body = await request.json()
    except Exception as e:
        return PlainTextResponse(f"Invalid JSON: {e}", status_code=400)

    result = await metrics_predict_logic(body)
    
    if result.payload is not None:
        return JSONResponse(result.payload, status_code=result.status, headers=result.headers)

    if result.text is not None:
        return PlainTextResponse(result.text, status_code=result.status, headers=result.headers)

    return Response(b"", status_code=result.status, headers=result.headers)


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
