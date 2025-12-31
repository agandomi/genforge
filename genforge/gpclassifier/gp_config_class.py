# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from __future__ import annotations

# This module is now a *wiring* layer:
# - Build a config object (from legacy flat params)
# - Resolve (validate + broadcast) via utils
# - Emit non-fatal diagnostics via gp.warning
# - Assign gp.config and gp.userdata

from .gp_config_data_class import ClassifierConfig
from .gp_config_utils_class import resolve_classifier_config

def gp_config(gp):
    """
    Validate and materialize gp.config and gp.userdata using the new config schema.
    Prefer gp.cfg (dataclass). Fall back to legacy gp.parameters only if present.
    """
    cfg = getattr(gp, "cfg", None)
    if cfg is None:
        # Back-compat: only if someone instantiates with legacy parameters
        legacy = getattr(gp, "parameters", {}) or {}
        cfg = ClassifierConfig.from_legacy(legacy)

    config, userdata, diags = resolve_classifier_config(gp, cfg, return_diagnostics=True)

    for msg in diags.warnings:
        gp.warning(msg)

    gp.config = config
    gp.userdata = userdata
    gp.cfg = cfg  # keep the resolved config snapshot for introspection
