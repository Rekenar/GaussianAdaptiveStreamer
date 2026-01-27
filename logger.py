import logging


logger = None

def getLogger():
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger("gs")
    logger.setLevel(logging.INFO)
    
    return logger


if(logger == None):
    logger = getLogger()


logger.info("Logger initialized!")