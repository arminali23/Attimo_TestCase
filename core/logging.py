import logging
from core.config import settings

def setup_logger(name: str = "rag_app") -> logging.Logger:
    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger