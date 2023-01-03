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
This module provides functions to optimize the cost volume using the 3SGM method

"""
import copy
import logging
import sys
from typing import Union, Tuple

import numpy as np
import xarray as xr

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

    def __init__(self, **cfg: Union[str, int, float, bool]):
        """
        :param cfg: optional configuration, {}
        :type cfg: dict
        :return: None
        """
        super().__init__(**cfg)
        self.cfg = self.check_geometric_prior(cfg)
        self._geometric_prior = self.cfg["geometric_prior"]

    def check_geometric_prior(self, cfg: dict) -> dict:
        """
        Verify geometric_prior parameter for 3sgm classification
        :param cfg:
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype cfg: dict
        """

        if "geometric_prior" not in cfg:
            cfg["geometric_prior"] = self._GEOMETRIC_PRIOR
        elif isinstance(cfg["geometric_prior"], dict):
            if not cfg["geometric_prior"]["source"] in self._AVAILABLE_GEOMETRIC_PRIOR:
                logging.error(f"{cfg['geometric_prior']['source']} is not available as a geometric prior")

        return cfg

    def desc(self):
        """
        Describes the optimization method
        """
        print("Optimization with SEGSEMSGM")

    def optimize_cv(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume with the 3SGM method

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

        # Apply get piecewise optimization layer array
        cv, geometric_prior_array = self.compute_piecewise_layer(img_left, self._geometric_prior, cv)  # type:ignore

        if self._sgm_version == "c++":
            cost_volumes_out = self.sgm_cpp(
                cv, invalid_value, confidence_is_int, p1_mat, p2_mat, geometric_prior_array, invalid_disp
            )

        else:
            run_sgm = self._method[0]
            cost_volumes_out = run_sgm(
                cv["cost_volume"].data,
                p1_mat,
                p2_mat,
                self._directions,
                geometric_prior_array,
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
        cv.attrs["optimization"] = "3sgm"

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

    @staticmethod
    def compute_piecewise_layer(
        img_left: xr.Dataset, geometric_prior: dict, cv: xr.Dataset
    ) -> Tuple[xr.Dataset, np.ndarray]:
        """
        Compute the piecewise optimization layer array to use in optimization

        :param img_left: left image dataset with the data variables:

                - im : 2D (row, col) xarray.DataArray float32
                - msk : 2D (row, col) xarray.DataArray int16, with the convention defined in the configuration file
                - classif (optional): 2D (row, col) xarray.DataArray
                - segm (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray.Dataset
        :param geometric_prior: Layer to use
        :type geometric_prior: str
        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :return: the piecewise optimization layer array and the cost_volume
        :rtype: Tuple[xr.Dataset, np.ndarray]
        """
        nb_rows, nb_cols = img_left["im"].data.shape
        # internal (from cv), segm or classif (from image)
        mode = geometric_prior["source"]

        if mode in ["segm", "classif"]:
            # if geometric_prior comes from the image (segm or classif)
            if mode in img_left:
                geometric_prior_array = img_left[mode].data
            else:
                logging.warning("%s not in image dataset.", mode)
                sys.exit(1)
        # if user wants to use another type of geometric prior
        else:
            geometric_prior_array = np.ones(img_left["im"].data.shape)
            if mode != "internal":
                logging.warning(
                    "User wants to use a mode not in image dataset. \n "
                    "Default is used : no optimization with 3sgm will be performed."
                )
            else:
                # if layer not computed we add a default one
                if "internal" not in cv:
                    logging.warning("For now, 3SGM doesn't compute piecewise layer from internal mode.")
                    prior_array = xr.DataArray(
                        data=geometric_prior_array,
                        coords=[("row", np.arange(nb_rows)), ("col", np.arange(nb_cols))],
                    )
                    logging.warning(
                        "User wants to use a mode not in cost volume. \n "
                        "Default is used : no optimization with 3sgm will be performed."
                    )
                    # Apply geometric prior to cost volume
                    cv["internal"] = prior_array
                else:
                    geometric_prior_array = cv["internal"].data

        geometric_prior_array = geometric_prior_array.astype(np.float32)
        geometric_prior_array[np.isnan(geometric_prior_array)] = -9999

        return cv, geometric_prior_array
