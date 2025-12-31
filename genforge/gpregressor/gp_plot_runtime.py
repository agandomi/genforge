# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

def setup_matplotlib_backend(live: bool = False, preferred: str | None = None) -> str:
    """
    Select a matplotlib backend once, before importing pyplot anywhere.
    Returns the chosen backend name. If GUI isn't available, falls back to Agg and disables live.
    """
    import os
    import matplotlib

    if not live:
        matplotlib.use("Agg", force=True)
        return "Agg"

    # Try explicit preference first
    candidates = []
    if preferred and preferred.lower() != "auto":
        candidates = [preferred]
    else:
        # Reasonable GUI backends to try in order
        candidates = ["QtAgg", "TkAgg", "MacOSX"]

    for cand in candidates:
        try:
            matplotlib.use(cand, force=True)
            return cand
        except Exception:
            continue

    # Fallback: headless
    matplotlib.use("Agg", force=True)
    return "Agg"
