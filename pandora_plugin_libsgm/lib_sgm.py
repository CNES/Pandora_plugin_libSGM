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
This module provides functions to optimize the cost volume using the LibSGM library
"""

import copy

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

    def optimize_cv(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume with the SGM method

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right: xarray
        :return: the optimize cost volume with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """
        invalid_disp = np.isnan(cv["cost_volume"].data)

        # libSGM uses dissimilarities : methods with similarity have to be multiplied by -1
        if cv.attrs["type_measure"] == "max":
            cv["cost_volume"].data *= -1

        img_left["im"].data = np.ascontiguousarray(img_left["im"].data, dtype=np.float32)
        img_right["im"].data = np.ascontiguousarray(img_right["im"].data, dtype=np.float32)

        # Compute penalties
        invalid_value, p1_mat, p2_mat = self._penalty.compute_penalty(cv, img_left, img_right)

        # Apply confidence to cost volume
        cv, confidence_is_int = self.apply_confidence(cv, self._use_confidence)  # type:ignore

        # Default optimization layer, for a piecewise optimization layer array use 3sgm method
        optimization_layer = np.ones(img_left["im"].data.shape, dtype=np.float32)

        if self._sgm_version == "c++":
            cost_volumes_out = self.sgm_cpp(
                cv, invalid_value, confidence_is_int, p1_mat, p2_mat, optimization_layer, invalid_disp
            )

        else:
            run_sgm = self._method[0]
            cost_volumes_out = run_sgm(
                cv["cost_volume"].data,
                p1_mat,
                p2_mat,
                self._directions,
                optimization_layer,
                cost_paths=self._min_cost_paths,
                overcounting=self._overcounting,
            )

        cv["cost_volume"].data = cost_volumes_out["cv"]

        # Allocate the number of paths given the min costs
        if self._min_cost_paths:
            cv = self.number_of_disp(cv, cost_volumes_out["cv_min"], invalid_disp)

        # The cost volume has to be multiplied by -1 to be re-considered as a similarity measure
        if cv.attrs["type_measure"] == "max":
            cv["cost_volume"].data *= -1

        if self._sgm_version == "c++":
            cv["cost_volume"].data[invalid_disp] = np.nan
        cv.attrs["optimization"] = "sgm"

        # add lr cost volumes if they exist
        for i in range(32):
            if "cv_" + repr(i) in cost_volumes_out:
                cv["cost_volume" + repr(i)] = copy.deepcopy(cv["cost_volume"])
                cv["cost_volume" + repr(i)].data = copy.deepcopy(cost_volumes_out["cv_" + repr(i)])

        # Remove temporary values
        del p1_mat, p2_mat, invalid_disp

        # Maximal cost of the cost volume after optimization
        cmax = invalid_value - 1
        cv.attrs["cmax"] = cmax

        return cv
