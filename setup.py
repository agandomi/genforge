# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi
from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

def read_version():
    init = (ROOT / "genforge" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.M)
    if not m:
        raise RuntimeError("Cannot find __version__ in genforge/__init__.py")
    return m.group(1)

def read_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        return []
    return [
        line.strip()
        for line in req.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

setup(
    name="genforge",
    version=read_version(),
    description="GenForge: Sculpting Solutions with Multi-Population Genetic Programming",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Mohammad Sadegh Khorshidi",
    author_email="msadegh.khorshidi.ak@gmail.com",
    license="GPL-3.0-only",
    url="https://github.com/maisamkhorshidi/genforge",
    packages=find_packages(include=["genforge*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


