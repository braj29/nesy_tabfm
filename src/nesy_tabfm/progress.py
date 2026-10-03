"""Run logging + progress bars shared by every audit entrypoint.

``setup_run_logging(out_dir)`` sends timestamped log lines to the console and to
``<out_dir>/run.log``; ``track(iterable, desc)`` wraps a loop in a tqdm bar. Both are safe to
call repeatedly and cost nothing when nobody is watching (tqdm writes to stderr).
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

LOGGER_NAME = "nesy_tabfm"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_run_logging(out_dir: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Console handler always; file handler at ``<out_dir>/run.log`` (appended) if given."""
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / "run.log", mode="a")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)
    return logger


def track(iterable: Iterable[T], desc: str, total: int | None = None) -> Iterator[T]:
    """tqdm bar over ``iterable`` (leave=True so finished bars stay visible in the scrollback)."""
    return tqdm(iterable, desc=desc, total=total, leave=True, dynamic_ncols=True)


@contextmanager
def timed(message: str):
    """Log ``message`` on entry and ``message (Xs)`` on exit; also logs the exception on failure."""
    logger = get_logger()
    logger.info("START %s", message)
    t0 = time.time()
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.1fs", message, time.time() - t0)
        raise
    logger.info("DONE  %s (%.1fs)", message, time.time() - t0)
