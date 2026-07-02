"""Simple logging utility."""
import logging
import sys
from pathlib import Path
from ..core.config import BASE_DIR, CFG

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    log_cfg = CFG.get("log", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper())
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        # File handler
        fh = logging.FileHandler(LOG_DIR / "audit.log")
        fh.setFormatter(formatter)
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger