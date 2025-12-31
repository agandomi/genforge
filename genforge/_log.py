# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from __future__ import annotations

import logging
import os
import functools
from typing import Optional, Callable

# -------- core logger helpers --------

def get_logger(name: str = "genforge") -> logging.Logger:
    """
    Return a logger configured with a stderr StreamHandler if none exist.
    We don't cache a single logger globally so different names can coexist.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()  # stderr
        f = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        h.setFormatter(f)
        logger.addHandler(h)
    logger.propagate = False  # don't bubble to root
    return logger


def set_level(logger: logging.Logger, level: str | int) -> None:
    """Set level on the logger and its handlers."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)


def add_file_handler(
    logger: logging.Logger,
    filepath: str,
    mode: str = "a",
    delay: bool = True,
) -> logging.Handler:
    """
    Attach (or reuse) a FileHandler for `filepath`. Returns the handler so
    callers can close/remove it later. Uses `delay=True` so the OS file isn't
    opened until the first emit.
    """
    abspath = os.path.abspath(filepath)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abspath:
            return h
    fh = logging.FileHandler(abspath, mode=mode, encoding="utf-8", delay=delay)
    fh.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s"))
    logger.addHandler(fh)
    return fh


def close_handler(logger: logging.Logger, handler: Optional[logging.Handler]) -> None:
    """Flush, close, and detach a single handler (safe on Windows so files can be moved/deleted)."""
    if handler is None:
        return
    try:
        handler.flush()
    finally:
        try:
            handler.close()
        finally:
            try:
                logger.removeHandler(handler)
            except Exception:
                pass


# -------- optional plumbing --------

def capture_external_warnings(enable: bool) -> None:
    """
    Forward *all* Python warnings to the logging system when True.
    Keep False by default so third-party libs (matplotlib/sklearn) follow
    the user's existing warning filters.
    """
    logging.captureWarnings(bool(enable))


def log_exceptions(logger_attr: str = "logger") -> Callable:
    """
    Decorator: log full traceback on uncaught exceptions, then re-raise.
    Looks up `self.<logger_attr>`; falls back to a package logger if missing.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception:
                logger = getattr(self, logger_attr, None) or get_logger("genforge")
                # .exception logs traceback from the active exception
                logger.exception(f"Uncaught error in {self.__class__.__name__}.{fn.__name__}()")
                raise
        return wrapper
    return deco
