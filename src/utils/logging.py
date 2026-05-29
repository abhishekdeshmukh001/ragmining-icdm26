"""Logger setup with stdout + optional file handler."""
import logging
import sys
from pathlib import Path


def get_logger(name="evistate", log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured; optionally attach a new file handler.
        if log_file:
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler) and getattr(h, "_path", None) == str(log_file):
                    return logger
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            fh._path = str(log_file)  # type: ignore[attr-defined]
            logger.addHandler(fh)
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh._path = str(log_file)  # type: ignore[attr-defined]
        logger.addHandler(fh)
    return logger
