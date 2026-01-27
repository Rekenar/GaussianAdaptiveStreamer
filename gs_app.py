# gs_app.py

'''
python http3_server.py --certificate certificates/ssl_cert.pem --private-key certificates/ssl_key.pem --host 0.0.0.0
'''
'''
 google-chrome \
  --enable-experimental-web-platform-features \
  --ignore-certificate-errors-spki-list=BSQJ0jkQ7wwhR7KvPZ+DSNk2XTZ/MS6xCbo9qu++VdQ= \
  --origin-to-force-quic-on=localhost:4433 \
  https://localhost:4433/models-ui
'''


from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles


from statics import STATIC_DIR
import routes

    
starlette = Starlette(
    routes=[
        Route("/", routes.render_handler, methods=["GET"]),
        Route("/render", routes.render_handler, methods=["POST"]),
        Route("/metrics/predict", routes.metrics_predict, methods=["POST"]),
        Route("/export", routes.export_experiment, methods=["GET"]),
        Route("/models", routes.get_list_of_all_available_models, methods=["GET"]), 
        Route("/models-ui", routes.models_page, methods=["GET"]),
        Route("/player", routes.player_page, methods=["GET"]),
        Route("/loadModel", routes.load_model, methods=["POST"]),
        Route("/movement", routes.save_movements, methods=["POST"]),
        Route("/saveImages", routes.save_images, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    ]
)

# the callable the aioquic server imports
async def app(scope, receive, send):
    await starlette(scope, receive, send)
    
    

