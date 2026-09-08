"""Structured logging for COGNIX."""
import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def setup_logging(level: str = "INFO", use_rich: bool = True) -> None:
    """Configure COGNIX-wide logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if use_rich and _RICH_AVAILABLE:
        handlers.append(RichHandler(rich_tracebacks=True, markup=True))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(handler)
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a named logger for a COGNIX module."""
    logger = logging.getLogger(f"cognix.{name}")
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
