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
This module provides functions to optimize the cost volume using the LibSGM library
"""

from typing import Tuple

import numpy as np
import xarray as xr

from . import abstract_sgm


@abstract_sgm.AbstractSGM.register_subclass("sgm")
class SGM(abstract_sgm.AbstractSGM):
    """

    SGM class is a plugin that allow to perform the optimization step by calling the LibSGM library

    """

    def desc(self):
        """
        Describes the optimization method
        """
        print("Optimization with SGM")

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

        # Default optimization layer, for a piecewise optimization layer array use 3sgm method
        optimization_layer = np.ones(img_shape, dtype=np.float32)

        return optimization_layer
