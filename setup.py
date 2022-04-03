#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("autopycoin", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "autopycoin"
DESCRIPTION = "Deep learning models for forecasting purposes."
with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
MAINTAINER = "G. Dubuc"
MAINTAINER_EMAIL = "gaet.dub1@gmail.com"
URL = "https://github.com/GaetanDu/autopycoin"
LICENSE = "Apache license 2.0"
DOWNLOAD_URL = "https://github.com/GaetanDu/autopycoin"
VERSION = __version__
INSTALL_REQUIRES = [
    "numpy",
    "matplotlib",
    "pandas",
    "tensorflow >= 2.8.0",
    "tensorflow-probability",
    "plotly",
    "keras-tuner",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.9",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": [
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "numpydoc",
        "matplotlib",
        "pydata_sphinx_theme",
    ],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
