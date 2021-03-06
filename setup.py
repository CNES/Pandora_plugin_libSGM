#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Pandora plugin LibSGM
#
#     https://github.com/CNES/Pandora_plugin_libsgm
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains the required libraries and softwares allowing to execute the software,
 and setup elements to configure and identify the software.
"""

from codecs import open as copen

from setuptools import setup, find_packages

cmdclass = {}
try:
    from sphinx.setup_command import BuildDoc

    cmdclass["build_sphinx"] = BuildDoc
except ImportError:
    print("WARNING: sphinx not available. Doc cannot be built")

REQUIREMENTS = ["numpy", "xarray", "json-checker", "rasterio", "libsgm==0.4.*", "pandora>=1.1.0"]

SETUP_REQUIREMENTS = ["setuptools-scm"]

REQUIREMENTS_DEV = {
    "dev": [
        "sphinx",
        "sphinx_rtd_theme",
        "sphinx_autoapi",
        "pytest",
        "pytest-cov",
        "pylint",
        "pre-commit",
        "mypy",
        "black",
    ]
}


def readme():
    with copen("README.md", "r", "utf-8") as fstream:
        return fstream.read()


setup(
    name="pandora_plugin_libsgm",
    use_scm_version=True,
    description="Pandora plugin to optimize the cost volume with the LibSGM library",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/CNES/Pandora_plugin_libsgm",
    author="CNES",
    author_email="myriam.cournet@cnes.fr",
    license="Apache License 2.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
    extras_require=REQUIREMENTS_DEV,
    entry_points="""
          [pandora.plugin]
          pandora_plugin_libsgm = pandora_plugin_libsgm.lib_sgm:SGM
      """,
    cmdclass=cmdclass,
    command_options={
        "build_sphinx": {
            "build_dir": ("setup.py", "doc/build/"),
            "source_dir": ("setup.py", "doc/source/"),
            "warning_is_error": ("setup.py", True),
        }
    },
)
