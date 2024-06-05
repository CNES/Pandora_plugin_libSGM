#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module provides functions to optimize the cost volume using the 3SGM method

"""

import logging
import sys
from typing import Union, Tuple

import numpy as np
import xarray as xr
from pandora.img_tools import fuse_classification_bands
from . import abstract_sgm


@abstract_sgm.AbstractSGM.register_subclass("3sgm")
class SEGSEMSGM(abstract_sgm.AbstractSGM):
    """

    SEGSEMSGM class is a plugin that allow to perform the optimization step
    by calling the LibSGM library with an optimization layer.

    """

    # Default configuration, do not change these values
    _GEOMETRIC_PRIOR = {"source": "internal"}
    _AVAILABLE_GEOMETRIC_PRIOR = ["internal", "classif", "segm"]

    def __init__(self, img: xr.Dataset, **cfg: Union[str, int, float, bool]):
        """
        :param img: xarray.Dataset of image with metadata
        :type img: xarray.Dataset
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """
        super().__init__(img, **cfg)
        self.cfg = self.check_geometric_prior(cfg)
        self._geometric_prior = self.cfg["geometric_prior"]
        if self._geometric_prior["source"] == "classif":  # type: ignore
            self._classes = self._geometric_prior["classes"]  # type: ignore

    def check_geometric_prior(self, cfg: dict) -> dict:
        """
        Verify geometric_prior parameter for 3sgm classification
        :param cfg:
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype cfg: dict
        """

        # Geometric prior is internal if not mentioned
        if "geometric_prior" not in cfg:
            cfg["geometric_prior"] = self._GEOMETRIC_PRIOR
        elif isinstance(cfg["geometric_prior"], dict):
            if not cfg["geometric_prior"]["source"] in self._AVAILABLE_GEOMETRIC_PRIOR:
                logging.error("%s is not available as a geometric prior", cfg["geometric_prior"]["source"])

        # Must be classif, segm or internal
        if cfg["geometric_prior"]["source"] not in self._AVAILABLE_GEOMETRIC_PRIOR:
            logging.error("%s is not available as a geometric prior source", cfg["geometric_prior"]["source"])
            sys.exit(1)

        # If source is classif, classes must be instantiated
        if cfg["geometric_prior"]["source"] == "classif":
            if "classes" not in cfg["geometric_prior"]:
                logging.error("Classes must be instantiated if source is classif")
                sys.exit(1)
            else:
                # If source is classif, classes must be instantiated as a list of string
                if not isinstance(cfg["geometric_prior"]["classes"], list) and not all(
                    isinstance(item, str) for item in cfg["geometric_prior"]["classes"]
                ):
                    logging.error("Classes must be a list of string")
                    sys.exit(1)

        if cfg["geometric_prior"]["source"] == "segm":
            if "classes" in cfg["geometric_prior"]:
                logging.error("Classes can't be instantiated if source is segm")
                sys.exit(1)

        return cfg

    def desc(self):
        """
        Describes the optimization method
        """
        print("Optimization with 3SGM")

    def compute_optimization_layer(
        self, cv: xr.Dataset, img_left: xr.Dataset, img_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Compute optimization layer for optimization method

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :param img_shape: shape of the input image
        :type img_shape: Tuple[int, ...]
        :return: the optimization layer array
        :rtype: np.ndarray
        """
        # internal (from cv), segm or classif (from image)
        mode = self._geometric_prior["source"]  # type: ignore

        if mode in ["segm", "classif"]:
            # if geometric_prior comes from the image (segm or classif)
            if mode in img_left:
                geometric_prior_array = img_left[mode].data
                if mode == "classif":
                    geometric_prior_array = fuse_classification_bands(img_left, self._classes)
            else:
                logging.warning("%s not in image dataset.", mode)
                sys.exit(1)
        # if user wants to use another type of geometric prior
        else:
            geometric_prior_array = np.ones(img_shape)
            if mode != "internal":
                logging.warning(
                    "User wants to use a mode not in image dataset. \n "
                    "Default is used : no optimization with 3sgm will be performed."
                )
            else:
                if "internal" not in cv:
                    # if layer not computed we add a default one
                    prior_array = xr.DataArray(
                        data=geometric_prior_array,
                        coords=[("row", np.arange(img_shape[0])), ("col", np.arange(img_shape[1]))],
                    )
                    logging.warning(
                        "User wants to use a mode not in cost volume. \n "
                        "Default is used : no optimization with 3sgm will be performed."
                    )
                    logging.warning(
                        "If semantic_segmentation is present in the pipeline, \n "
                        "it should be placed before the 3SGM optimization step."
                    )
                    # Apply geometric prior to cost volume
                    cv["internal"] = prior_array
                else:
                    geometric_prior_array = cv["internal"].data

        geometric_prior_array = geometric_prior_array.astype(np.float32)
        geometric_prior_array[np.isnan(geometric_prior_array)] = -9999

        return geometric_prior_array
