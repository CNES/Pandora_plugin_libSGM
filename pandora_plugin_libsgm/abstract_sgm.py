#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains classes and functions associated to the cost volume optimization step.
"""

import copy
import logging
import sys
from abc import abstractmethod
from typing import Dict, Union, Tuple, Optional, cast

import numpy as np
import xarray as xr
from json_checker import Checker, And, OptionalKey
from libSGM import sgm_wrapper
from pandora.common import is_method
from pandora.cost_volume_confidence import AbstractCostVolumeConfidence
from pandora.optimization import optimization

from pandora_plugin_libsgm import penalty

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


class AbstractSGM(optimization.AbstractOptimization):
    """
    AbstractSGM input class
    """

    # Default configuration, do not change these values
    _SGM_VERSION = "c++"
    _OVERCOUNTING = False
    _MIN_COST_PATH = False
    _PENALTY = {"penalty_method": "sgm_penalty", "P1": 4, "P2": 20}
    _DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1], [0, -1], [-1, 0], [-1, -1], [-1, 1]]
    _USE_CONFIDENCE = None

    def __init__(self, img: xr.Dataset, **cfg: Union[str, int, float, bool, dict]):
        """
        :param img: xarray.Dataset of image with metadata
        :type img: xarray.Dataset
        :param cfg: optional configuration, {'P1': value, 'P2': value, 'alpha': value, 'beta': value, 'gamma': value,
                                            'p2_method': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(img, **cfg)
        self._sgm_version = cast(str, self.cfg["sgm_version"])
        self._overcounting = self.cfg["overcounting"]
        self._min_cost_paths = self.cfg["min_cost_paths"]
        self._use_confidence = self.cfg.get("use_confidence", self._USE_CONFIDENCE)
        self._directions = self._DIRECTIONS
        self._penalty = penalty.AbstractPenalty(self._directions, **self.cfg["penalty"])  # type: ignore

        # Get Python versions of LibSGM
        self._method = []
        for entry_point in entry_points(group="libsgm", name=self._sgm_version):
            self._method.append(entry_point.load())

    def desc(self):
        """
        Describes the optimization method
        """

    def check_conf(
        self, img: xr.Dataset, **cfg: Union[str, int, float, bool, dict]
    ) -> Dict[str, Union[str, int, float, bool, dict]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param img: xarray.Dataset of image with metadata
        :type img: xarray.Dataset
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
        if "penalty" not in cfg:
            cfg["penalty"] = self._PENALTY
        if cfg["optimization_method"] == "sgm" and "geometric_prior" in cfg:
            logging.error("Geometric prior not available for SGM optimization")
            sys.exit(1)

        if "geometric_prior" in cfg:
            source = cfg["geometric_prior"]["source"]  # type: ignore[index]
            if source in ["classif", "segm"] and not source in img.data_vars:
                logging.error(
                    "For performing the 3SGM optimization step in the pipeline, left %s must be present.", source
                )
                sys.exit(1)

        schema = {
            "sgm_version": And(str, lambda x: is_method(x, ["c++", "python_libsgm", "python_libsgm_parall"])),
            "optimization_method": And(str, lambda x: is_method(x, ["sgm", "3sgm"])),
            "overcounting": bool,
            "min_cost_paths": bool,
            OptionalKey("use_confidence"): str,
            "penalty": dict,
            OptionalKey("geometric_prior"): dict,
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def optimize_cv(self, cv: xr.Dataset, img_left: xr.Dataset, img_right: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume with the SGM method

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_left: xarray
        :param img_right: right Dataset image containing :

                - im : 2D (row, col) or 3D (band, row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :type img_right: xarray
        :return: the optimized cost volume with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """

        invalid_disp = np.isnan(cv["cost_volume"].data)

        # libSGM uses dissimilarities : methods with similarity have to be multiplied by -1
        if cv.attrs["type_measure"] == "max":
            cv["cost_volume"].data *= -1

        # If the input images were multiband, the band used for the correlation is used
        img_left_array = get_band_values(img_left, cv.attrs["band_correl"])
        img_right_array = get_band_values(img_right, cv.attrs["band_correl"])

        # Compute penalties
        invalid_value, p1_mat, p2_mat = self._penalty.compute_penalty(cv, img_left_array, img_right_array)

        # Apply confidence to cost volume
        cv, confidence_is_int = self.apply_confidence(cv, self._use_confidence)  # type:ignore

        # get optimization layer and add optimization layer to cost volume if necessary
        optimization_layer = self.compute_optimization_layer(cv, img_left, img_left_array.shape)

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
        cv.attrs["optimization"] = self.cfg["optimization_method"]

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

    @abstractmethod
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

    @staticmethod
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
            # To reduce memory, the cost volume is split (along the col axis)
            # into multiple sub-arrays with a step of 100
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

    def number_of_disp(self, cv: xr.Dataset, disp_paths: np.ndarray, invalid_disp: np.ndarray) -> xr.Dataset:
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
        disp_map = self.argmin_split(cv)
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
            "optimization_plugin_libsgm_nb_of_directions", conf_map, None, cv
        )

        del invalid_mc
        del disp_map

        return cv

    @staticmethod
    def apply_confidence(cv: xr.Dataset, use_confidence: str) -> Tuple[xr.Dataset, bool]:
        """
        Apply the confidence measure to cost volume,as weights.

        :param cv: the original cost volume dataset with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param use_confidence: Apply or not confidence
        :type use_confidence: str
        :return: the cost volume dataset updated with a new indicator with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :rtype: xarray.Dataset
        """

        nb_rows, nb_cols, _ = cv["cost_volume"].data.shape
        # Initialise confidence ( in [0, 1])
        confidence_is_int = True
        if use_confidence is not None:
            measure_coord = "confidence_from_ambiguity"
            suffix_exists = use_confidence.find('.')
            if suffix_exists >= 0:
                measure_coord += use_confidence[suffix_exists:]
            if "confidence_measure" in cv and measure_coord in cv.coords["indicator"]:
                confidence_is_int = False
                confidence_array = cv["confidence_measure"].sel(indicator=measure_coord).data
            else:
                confidence_array = np.ones((nb_rows, nb_cols))
                logging.warning(
                    "User wants to use %s that was not computed previously or an ambiguity confidence \n "
                    "Default is used : confidence values will be equal to 1, which is equivalent to not use \n "
                    "confidence.",
                    use_confidence
                )
        else:
            confidence_array = np.ones((nb_rows, nb_cols))

        # Apply confidence to cost volume
        confidence_array[np.isnan(confidence_array)] = 1
        cv["cost_volume"].data *= np.expand_dims(confidence_array, axis=2)
        cv["cost_volume"].data = cv["cost_volume"].data.astype(np.float32)

        return cv, confidence_is_int

    def sgm_cpp(
        self,
        cv: xr.Dataset,
        invalid_value: float,
        confidence_is_int: bool,
        p1_mat: np.ndarray,
        p2_mat: np.ndarray,
        optim_layer: np.ndarray,
        invalid_disp: np.ndarray,
    ):
        """
        Compute aggregated cost volume using C++ library where sgm method is implemented

        :param cv: the cost volume, with the data variables:

                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param invalid_value: invalid value for penalties
        :type invalid_value: float
        :param confidence_is_int: confidence is int
        :type confidence_is_int: bool
        :param p1_mat: P1 penalties
        :type p1_mat: np.array
        :param p2_mat: P2 penalties
        :type p2_mat: np.array
        :param optim_layer: optimization layer
        :type optim_layer: np.array
        :param invalid_disp: invalid disparities mask
        :type invalid_disp: np.array
        """
        # If the cost volume is calculated with the census measure and the invalid value <= 255,
        # the cost volume is converted to unint8 to optimize the memory
        # Invalid value must not exceed the maximum value of uint8 type (255)
        if cv.attrs["measure"] == "census" and invalid_value <= 255 and not confidence_is_int:
            invalid_value = int(invalid_value)
            cost_volume = np.nan_to_num(cv["cost_volume"].data, nan=invalid_value)
            cv["cost_volume"].data = np.floor(cost_volume).astype(np.uint8)

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
            optim_layer,
            self._min_cost_paths,
            self._overcounting,
        )

        return cost_volumes_out


def get_band_values(image_dataset: xr.Dataset, band_name: Optional[str] = None) -> np.ndarray:
    """
    Get values of given band_name from image_dataset as numpy array.

    if band_name is not provided or is None, returns all bands values.

    :param image_dataset: dataset to extract data from.
    :param band_name: band_name to extract. If None selects all bands.
    :return: selected values.
    """
    selection = image_dataset if band_name is None else image_dataset.sel(band_im=band_name)
    return selection["im"].to_numpy()
