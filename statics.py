import os


ROOT = os.path.dirname(__file__)


CAPTURES_DIR = os.path.join(ROOT, "captures")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiment")
TEMPLATES_DIR = os.path.join(ROOT, "templates")
LOG_DIR = os.path.join(ROOT, "logs")
STATIC_DIR = os.path.join(ROOT, "static")
MODELS_DIR = os.path.join(STATIC_DIR, "models")

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)



PREVIEW_CANDIDATES = ("preview.png", "preview.jpg")

EVICT_AFTER_MS = 10 * 60_000
EVICT_CHECK_EVERY_S = 20