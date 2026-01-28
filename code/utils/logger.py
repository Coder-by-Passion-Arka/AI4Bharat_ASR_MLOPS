"""
Reusable logging utility for AI4Bharat ASR Optimization Pipeline.

Features:
- Console + file logging
- One log file per Python script
- Automatic script name detection
- Safe against duplicate handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str | None = None,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    log_file: Path | str | None = None,
    **kwargs,                       # accept extra kwargs to be backwards-compatible
):
    """
    Create or retrieve a logger.

    Backwards-compatible: accepts legacy 'log_file' kwarg.
    """
    # Infer script name automatically
    if name is None:
        script = Path(sys.argv[0]).stem or "interactive"
        name = script

    logger = logging.getLogger(name)

    # Prevent duplicate handlers (CRITICAL)
    if logger.handlers:
        logger.setLevel(level)
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Resolve project root (assumes logger.py is inside code/utils/)
    try:
        project_root = Path(__file__).resolve().parents[2]
    except Exception:
        project_root = Path.cwd()

    # Default log directory
    if log_dir is None:
        log_dir = project_root / "results" / "terminal_logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # If legacy single-file path provided via 'log_file', prefer it
    if log_file:
        # allow string or Path
        log_file = Path(log_file)
        # if a relative filename was provided, place inside log_dir
        if not log_file.is_absolute():
            log_file = (log_dir / log_file).resolve()
    else:
        log_file = log_dir / f"{name}.log"

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)  # always keep full detail in file
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized")
    logger.debug(f"Log file: {log_file}")

    return logger