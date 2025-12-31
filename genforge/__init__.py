# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from .gpclassifier.gp_class import gpclassifier
from .gpregressor.gp_regress import gpregressor
from .spfp.spfp_partition import SPFPPartitioner
from .gpclassifier.gp_config_data_class import ClassifierConfig
from .gpregressor.gp_config_data_regress import RegressorConfig


__all__ = ["gpclassifier", "gpregressor", "SPFPPartitioner", "RegressorConfig", "ClassifierConfig"]

__version__ = "1.0"

# GenForge software metadata
__title__ = "GenForge: Sculpting Solutions with Multi-Population Genetic Programming"
__author__ = "Mohammad Sadegh Khorshidi"
__author_email__ = "msadegh.khorshidi.ak@gmail.com"
__license__ = "GPL-3.0-only"
__url__ = "https://github.com/maisamkhorshidi/genforge"
__summary__ = __title__

# Structured object used by runtime info reporters
__package_metadata__ = {
    "name": "genforge",
    "title": __title__,
    "version": __version__,
    "license": __license__,
    "authors": [{"name": __author__, "email": __author_email__}],
    "url": __url__,
    "summary": __summary__,
}
