# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

from __future__ import annotations
import sys, platform, importlib
from typing import Dict, Any, Optional

def _lib_version(mod: str, attr: str = "__version__") -> Optional[str]:
    try:
        m = importlib.import_module(mod)
        return getattr(m, attr, None)
    except Exception:
        return None

def collect_package_info() -> Dict[str, Any]:
    # GenForge package metadata
    genforge = importlib.import_module("genforge")
    pkg_meta = getattr(genforge, "__package_metadata__", {
        "name": "genforge",
        "title": getattr(genforge, "__title__", "GenForge"),
        "version": getattr(genforge, "__version__", None),
        "license": getattr(genforge, "__license__", "GPL-3.0-only"),
    })

    # SPFP component metadata (if available)
    spfp_meta = {}
    try:
        spfp = importlib.import_module("genforge.spfp")
        spfp_meta = getattr(spfp, "__package_metadata__", {})
    except Exception:
        pass

    env = {
        "python_version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    libs = {
        "numpy": _lib_version("numpy"),
        "scikit_learn": _lib_version("sklearn"),
        "scipy": _lib_version("scipy"),
        "pandas": _lib_version("pandas"),
        "matplotlib": _lib_version("matplotlib"),
        "seaborn": _lib_version("seaborn"),
        "joblib": _lib_version("joblib"),
    }

    return {"package": pkg_meta, "components": {"spfp": spfp_meta}, "environment": env, "libraries": libs}
