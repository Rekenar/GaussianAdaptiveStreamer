from PIL import Image
import io
import numpy as np


def encode_jpeg(img: np.ndarray, *, quality: int = 70) -> bytes:
    im = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()

def encode_png(img: np.ndarray, *, compress_level: int = 6) -> bytes:
    if img.dtype != np.uint8:
        raise TypeError(f"PNG expects uint8, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"PNG expects HxWx3 RGB image, got shape {img.shape}")
    compress_level = int(np.clip(compress_level, 0, 9))
    im = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()
