import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured module logger.

    Parameters
    ----------
    name:
        Logger name, typically __name__.
    level:
        Optional log level override (e.g. "INFO", "DEBUG").
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    configured_level = level or os.getenv("ICU_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, configured_level, logging.INFO))
    logger.propagate = False
    return logger
