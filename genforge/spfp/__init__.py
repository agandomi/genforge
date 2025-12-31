# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

from .spfp_partition import SPFPPartitioner

__all__ = ["SPFPPartitioner"]

# SPFP component metadata 
__title__ = "SPFP: Semantic Preserving Feature Partitioning for Multi-View Learning"
__summary__ = __title__
__license__ = "GPL-3.0-only"
__url__ = "https://github.com/maisamkhorshidi/genforge"  # housed within GenForge
__version__ = "1.0"

# Author
__author__ = "Mohammad Sadegh Khorshidi"
__author_email__ = "msadegh.khorshidi.ak@gmail.com"

# Official citation 
__citation__ = (
    "Khorshidi, M. S., Yazdanjue, N., Gharoun, H., Yazdani, D., Nikoo, M. R., "
    "Chen, F., & Gandomi, A. H. (2025). Semantic-Preserving Feature Partitioning "
    "for multi-view ensemble learning. Information Fusion, 122, 103152. "
    "https://doi.org/10.1016/j.inffus.2025.103152"
)
__doi__ = "https://doi.org/10.1016/j.inffus.2025.103152"

# Structured object for programmatic access
__package_metadata__ = {
    "component": "spfp",
    "title": __title__,
    "summary": __summary__,
    "license": __license__,
    "url": __url__,
    "version": __version__,
    "authors": [{"name": __author__, "email": __author_email__}],
    "citation": __citation__,
    "doi": __doi__,
}
