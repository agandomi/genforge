# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

from .gp_class import gpclassifier
from .gp_config_data_class import ClassifierConfig

__title__ = "GenForge - GPClassifier"
__all__ = ["gpclassifier", "ClassifierConfig"]
__summary__ = "Symbolic regression with multi-population genetic programming for classification tasks"
__license__ = "GPL-3.0-only"
__url__ = "https://github.com/maisamkhorshidi/genforge"  # housed within GenForge
__version__ = "1.0"

# Author
__author__ = "Mohammad Sadegh Khorshidi"
__author_email__ = "msadegh.khorshidi.ak@gmail.com"

# Official citation 
__citation__ = (
    "Khorshidi, M. S., Yazdanjue, N., Gharoun, H., Nikoo, M. R., Chen, F., & Gandomi, A. H. (2025). "
    "Multi-population Ensemble Genetic Programming via Cooperative Coevolution and Multi-view Learning for Classification. "
    "arXiv preprint arXiv:2509.19339. https://doi.org/10.48550/arXiv.2509.19339\n\n"
    
    "Khorshidi, M. S., Yazdanjue, N., Gharoun, H., Nikoo, M. R., Chen, F., & Gandomi, A. H. (2025). "
    "From embeddings to equations: Genetic-programming surrogates for interpretable transformer classification. "
    "arXiv preprint arXiv:2509.21341. https://doi.org/10.48550/arXiv.2509.21341\n\n"
    
    "Khorshidi, M. S., Yazdanjue, N., Gharoun, H., Nikko, M. R., Chen, F., & Gandomi, A. H. (2025). "
    "Domain-Informed Genetic Superposition Programming: A Case Study on SFRC Beams. "
    "arXiv preprint arXiv:2509.21355. https://doi.org/10.48550/arXiv.2509.21355"
)

__doi__ = [
    "https://doi.org/10.48550/arXiv.2509.19339",
    "https://doi.org/10.48550/arXiv.2509.21341",
    "https://doi.org/10.48550/arXiv.2509.21355"
]


# Structured object for programmatic access
__package_metadata__ = {
    "component": "gpclassifer",
    "config_class": "ClassifierConfig",
    "title": __title__,
    "summary": __summary__,
    "license": __license__,
    "url": __url__,
    "version": __version__,
    "authors": [{"name": __author__, "email": __author_email__}],
    "citation": __citation__,
    "doi": __doi__,
}