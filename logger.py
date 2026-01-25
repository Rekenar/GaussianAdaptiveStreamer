import os
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler

from statics import LOG_DIR

logger = None

def getLogger():
    today = datetime.now().strftime("%Y-%m-%d")
    LOG_FILE = os.path.join(LOG_DIR, f"gs_server_{today}.log")

    logging.basicConfig(level=logging.INFO)

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
    
    return logger


if(logger == None):
    logger = getLogger()


logger.info("Logger initialized!")