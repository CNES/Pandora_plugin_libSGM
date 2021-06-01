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
import logging
from typing import Dict, Union, Tuple

import numpy as np
import xarray as xr
from libSGM import sgm_wrapper  # pylint: disable=no-name-in-module
from pandora.cost_volume_confidence import AbstractCostVolumeConfidence
from pandora.optimization import optimization
from pkg_resources import iter_entry_points

from pandora_plugin_libsgm import penalty


@optimization.AbstractOptimization.register_subclass("sgm")
class SGM(optimization.AbstractOptimization):
    """

    SGM class is a plugin that allow to perform the optimization step by calling the LibSGM library

    """

    # Default configuration, do not change these values
    _SGM_VERSION = "c++"
    _OVERCOUNTING = False
    _MIN_COST_PATH = False
    _PENALTY_METHOD = "sgm_penalty"
    _DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1], [0, -1], [-1, 0], [-1, -1], [-1, 1]]
    _USE_CONFIDENCE = False
    _PIECEWISE_OPTIMIZATION_LAYER = "None"

    def __init__(self, **cfg: Union[str, int, float, bool]):
        """
        :param cfg: optional configuration, {'P1': value, 'P2': value, 'alpha': value, 'beta': value, 'gamma': value,
                                            'p2_method': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._sgm_version = self.cfg["sgm_version"]
        self._overcounting = self.cfg["overcounting"]
        self._min_cost_paths = self.cfg["min_cost_paths"]
        self._use_confidence = self.cfg["use_confidence"]
        self._piecewise_optimization_layer = self.cfg["piecewise_optimization_layer"]
        self._directions = self._DIRECTIONS
        self._penalty = penalty.AbstractPenalty(self._directions, **self.cfg)  # type: ignore

        # Get Python versions of LibSGM
        self._method = []
        for entry_point in iter_entry_points(group="libsgm", name=self._sgm_version):  # type: ignore
            self._method.append(entry_point.load())

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: optimization configuration
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype cfg: dict
        """
        # Give the default value if the required element is not in the configuration
        if "sgm_version" not in cfg:
            cfg["sgm_version"] = self._SGM_VERSION
        if "overcounting" not in cfg:
            cfg["overcounting"] = self._OVERCOUNTING
        if "min_cost_paths" not in cfg:
            cfg["min_cost_paths"] = self._MIN_COST_PATH
        if "penalty_method" not in cfg:
            cfg["penalty_method"] = self._PENALTY_METHOD
        if "use_confidence" not in cfg:
            cfg["use_confidence"] = self._USE_CONFIDENCE
        if "piecewise_optimization_layer" not in cfg:
            cfg["piecewise_optimization_layer"] = self._PIECEWISE_OPTIMIZATION_LAYER

        return cfg

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

        # Apply get piecewise optimization layer array
        piecewise_optimization_layer_array = self.compute_piecewise_layer(
            img_left,
            self._piecewise_optimization_layer,  # type:ignore
        )

        if self._sgm_version == "c++":
            # If the cost volume is calculated with the census measure and the invalid value <= 255,
            # the cost volume is converted to unint8 to optimize the memory
            # Invalid value must not exceed the maximum value of uint8 type (255)
            if cv.attrs["measure"] == "census" and invalid_value <= 255 and not confidence_is_int:
                invalid_value = int(invalid_value)
                cv["cost_volume"].data = cv["cost_volume"].data.astype(np.uint8)

            p1_mat, p2_mat = (
                p1_mat.astype(cv["cost_volume"].data.dtype.type),
                p2_mat.astype(cv["cost_volume"].data.dtype.type),
            )

            # Conversion of invalid cost (= np.nan), to invalid_value
            cv["cost_volume"].data[invalid_disp] = invalid_value
            cv["cost_volume"].data = np.ascontiguousarray(cv["cost_volume"].data)

            # LibSGM library takes as input a numpy array, and output a numpy array
            cost_volumes_out = sgm_wrapper.sgm_api(
                cv["cost_volume"].data,
                p1_mat,
                p2_mat,
                np.array(self._directions).astype(np.int32),
                invalid_value,
                piecewise_optimization_layer_array,
                self._min_cost_paths,
                self._overcounting,
            )

        else:
            run_sgm = self._method[0]
            cost_volumes_out = run_sgm(
                cv["cost_volume"].data,
                p1_mat,
                p2_mat,
                self._directions,
                piecewise_optimization_layer_array,
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

    @staticmethod
    def number_of_disp(cv: xr.Dataset, disp_paths: np.ndarray, invalid_disp: np.ndarray) -> xr.Dataset:
        """
        Update the confidence measure by adding the number of disp indicator, which gives the number (between 0 and 8)
        of local disparities equal to the ones which return the global minimal costs

        :param cv: the original cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param disp_paths: the disparities given the minimum cost
        :type disp_paths: numpy.array
        :param invalid_disp: invalid pixels of the cost_volume
        :type invalid_disp: np.ndarray
        :return: the cost volume dataset updated with a new indicator with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """

        # Compute the disparity given the minimum cost volume for each pixel
        cv["cost_volume"].data[invalid_disp] = np.inf
        disp_map = argmin_split(cv)
        invalid_mc = np.min(invalid_disp, axis=2)
        invalid_pixel = np.where(invalid_mc)
        disp_map[invalid_pixel] = np.nan

        # Compute the confidence measure
        row, col, _ = cv["cost_volume"].shape
        conf_map = np.zeros((row, col), dtype=np.float32)

        # Allocate the number of paths given the same disparity as the one which has calculated the min cost
        for disp in range(disp_paths.shape[2]):
            pos_y, pos_x = np.where(disp_paths[:, :, disp] == disp_map)
            conf_map[pos_y, pos_x] += 1
        conf_map[invalid_pixel] = np.nan

        # Allocate the confidence measure
        _, cv = AbstractCostVolumeConfidence.allocate_confidence_map(
            "optimization_pluginlibSGM_nbOfDisp", conf_map, None, cv
        )

        del invalid_mc
        del disp_map

        return cv

    @staticmethod
    def apply_confidence(cv: xr.Dataset, use_confidence: bool) -> Tuple[xr.Dataset, bool]:
        """
        Apply the confidence measure to cost volume,as weights.

        :param cv: the original cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param use_confidence: Apply or not confidence
        :type use_confidence: bool
        :return: the cost volume dataset updated with a new indicator with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """

        nb_rows, nb_cols, _ = cv["cost_volume"].data.shape
        # Initialise confidence ( in [0, 1])
        confidence_is_int = True
        if use_confidence:
            if "confidence_measure" in cv and "ambiguity_confidence" in cv.coords["indicator"]:
                confidence_is_int = False
                confidence_array = cv["confidence_measure"].sel(indicator="ambiguity_confidence").data
            else:
                confidence_array = np.ones((nb_rows, nb_cols))
                logging.warning(
                    "User wants to use ambiguity confidence that was not computed previously \n "
                    "Default is used : confidence values will be equal to 1, which is equivalent to not use confidence."
                )
        else:
            confidence_array = np.ones((nb_rows, nb_cols))

        # Apply confidence to cost volume
        confidence_array[np.isnan(confidence_array)] = 1
        cv["cost_volume"].data *= np.expand_dims(confidence_array, axis=2)
        cv["cost_volume"].data = cv["cost_volume"].data.astype(np.float32)

        return cv, confidence_is_int

    @staticmethod
    def compute_piecewise_layer(img_left: xr.Dataset, piecewise_optimization_layer: str) -> np.ndarray:
        """
        Compute the piecewise optimization layer array to use in optimization

        :param img_left: left image dataset with the data variables:

                - im : 2D (row, col) xarray.DataArray float32
                - msk : 2D (row, col) xarray.DataArray int16, with the convention defined in the configuration file
                - classif (optional): 2D (row, col) xarray.DataArray
                - segm (optional): 2D (row, col) xarray.DataArray
        :type cv: xarray.Dataset
        :param piecewise_optimization_layer: Layer to use
        :type piecewise_optimization_layer: str
        :return: the piecewise optimization layer array
        :rtype: np.ndarray
        """

        if piecewise_optimization_layer in img_left:
            piecewise_optimization_layer_array = img_left[piecewise_optimization_layer].data
        else:
            piecewise_optimization_layer_array = np.ones(img_left["im"].data.shape)
            if piecewise_optimization_layer != "None":
                logging.warning(
                    "User wants to use a piecewise_optimization_layer not in image dataset. \n "
                    "Default is used : no piecewise optimization will be performed."
                )

        piecewise_optimization_layer_array = piecewise_optimization_layer_array.astype(np.float32)
        piecewise_optimization_layer_array[np.isnan(piecewise_optimization_layer_array)] = -9999

        return piecewise_optimization_layer_array


def argmin_split(cost_volume: xr.Dataset) -> np.ndarray:
    """
    Find the indices of the minimum values for a 3D DataArray, along axis 2.
    Memory consumption is reduced by splitting the 3D Array.

    Different from the argmax_split function of Pandora.Disparity because it returns numpy array argmin indexes
    and not relative indexes (relative: according to [dmin,dmax] range)

    :param cost_volume: the cost volume dataset
    :type cost_volume: xarray.Dataset
    :return: the disparities for which the cost volume values are the smallest
    :rtype: np.ndarray
    """
    ncol, nrow, _ = cost_volume["cost_volume"].shape
    disp = np.zeros((ncol, nrow), dtype=np.float32)

    # Numpy argmin is making a copy of the cost volume.
    # To reduce memory, numpy argmin is applied on a small part of the cost volume.
    # The cost volume is split (along the row axis) into multiple sub-arrays with a step of 100
    cv_chunked_y = np.array_split(cost_volume["cost_volume"].data, np.arange(100, ncol, 100), axis=0)

    y_begin = 0

    nb_chunks_y = len(cv_chunked_y)
    for num_chunk_y in range(nb_chunks_y):
        # To reduce memory, the cost volume is split (along the col axis) into multiple sub-arrays with a step of 100
        cv_chunked_x = np.array_split(cv_chunked_y[num_chunk_y], np.arange(100, nrow, 100), axis=1)
        x_begin = 0
        nb_chunks_x = len(cv_chunked_x)
        for num_chunk_x in range(nb_chunks_x):
            disp[
                y_begin : y_begin + cv_chunked_y[num_chunk_y].shape[0],
                x_begin : x_begin + cv_chunked_x[num_chunk_x].shape[1],
            ] = np.argmin(cv_chunked_x[num_chunk_x], axis=2)
            x_begin += cv_chunked_x[num_chunk_x].shape[1]

        y_begin += cv_chunked_y[num_chunk_y].shape[0]

    return disp
