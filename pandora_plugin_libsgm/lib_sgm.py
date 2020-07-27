#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of Pandora plugin LibSGM
#
#     https://github.com/CNES/Pandora_plugin_libsgm
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
This module provides functions to optimize the cost volume using the LibSGM library
"""

import numpy as np
import xarray as xr
from typing import Dict, Union

from pandora.JSON_checker import is_method
from pandora.optimization import optimization
from libSGM import sgm_wrapper
from pandora_plugin_libsgm import penalty
from pandora_plugin_libsgm import penalty_sgm
from pandora_plugin_libsgm import penalty_mc_cnn


@optimization.AbstractOptimization.register_subclass('sgm')
class SGM(optimization.AbstractOptimization):
    """

    SGM class is a plugin that allow to perform the optimization step by calling the LibSGM library

    """
    # Default configuration, do not change these values
    _OVERCOUNTING = False
    _MIN_COST_PATH = False
    _PENALTY_METHOD = "sgm_penalty"
    _DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1], [0, -1], [-1, 0], [-1, -1], [-1, 1]]

    def __init__(self, **cfg):
        """
        :param cfg: optional configuration, {'P1': value, 'P2': value, 'alpha': value, 'beta': value, 'gamma": value,
                                            'p2_method': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._overcounting = self.cfg['overcounting']
        self._min_cost_paths = self.cfg['min_cost_paths']
        self._directions = self._DIRECTIONS
        self._penalty = penalty.AbstractPenalty(self._directions, ** self.cfg)

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: optimization configuration
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype cfg: dict
        """
        # Give the default value if the required element is not in the configuration
        if 'overcounting' not in cfg:
            cfg['overcounting'] = self._OVERCOUNTING
        if 'min_cost_paths' not in cfg:
            cfg['min_cost_paths'] = self._MIN_COST_PATH
        if 'penalty_method' not in cfg:
            cfg['penalty_method'] = self._PENALTY_METHOD

        return cfg

    def desc(self):
        """
        Describes the optimization method
        """
        print('Optimization with SGM')

    def optimize_cv(self, cv: xr.Dataset, img_ref: xr.Dataset, img_sec: xr.Dataset) -> xr.Dataset:
        """
        Optimizes the cost volume with the SGM method

        :param cv: the cost volume
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param img_ref: reference Dataset image
        :type img_ref:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :param img_sec: secondary Dataset image
        :type img_sec:
            xarray.Dataset containing :
                - im : 2D (row, col) xarray.DataArray
                - msk (optional): 2D (row, col) xarray.DataArray
        :return: the optimize cost volume
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """
        invalid_disp = np.isnan(cv['cost_volume'].data)

        # libSGM uses dissimilarities : methods with similarity have to be multiplied by -1
        if cv.attrs['type_measure'] == "max":
            cv['cost_volume'].data *= -1

        # Resize pandora image : image size and cost volume size must be equal
        offset = int(cv.attrs['offset_row_col'])
        if offset == 0 :
            img_ref_crop = img_ref['im'].data
            img_sec_crop = img_sec['im'].data
        else :
            img_ref_crop = img_ref['im'].data[offset: -offset, offset: -offset]
            img_sec_crop = img_sec['im'].data[offset: -offset, offset: -offset]

        img_ref_crop = np.ascontiguousarray(img_ref_crop, dtype=np.float32)
        img_sec_crop = np.ascontiguousarray(img_sec_crop, dtype=np.float32)

        # Compute penalities
        invalid_value, p1_mat, p2_mat = self._penalty.compute_penalty(cv, img_ref_crop, img_sec_crop)

        # If the cost volume is calculated with the census measure and the invalid value <= 255,
        # the cost volume is converted to unint8 to optimize the memory
        # Invalid value must not exceed the maximum value of uint8 type (255)
        if cv.attrs['measure'] == "census" and invalid_value <= 255:
            invalid_value = int(invalid_value)
            cv['cost_volume'].data = cv['cost_volume'].data.astype(np.uint8)

        p1_mat, p2_mat = p1_mat.astype(cv['cost_volume'].data.dtype.type), p2_mat.astype(cv['cost_volume'].data.dtype.type)

        # Conversion of invalid cost (= np.nan), to invalid_value
        cv['cost_volume'].data[invalid_disp] = invalid_value
        cv['cost_volume'].data = np.ascontiguousarray(cv['cost_volume'].data)

        # LibSGM library takes as input a numpy array, and output a numpy array
        cost_volumes_out = sgm_wrapper.sgm_api(cv['cost_volume'].data, p1_mat, p2_mat,
                                               np.array(self._directions).astype(np.int32), invalid_value,
                                               self._min_cost_paths, self._overcounting)

        cv['cost_volume'].data = cost_volumes_out["cv"]

        # Allocate the number of paths given the min costs
        if self._min_cost_paths:
            cv = self.number_of_disp(cv, cost_volumes_out["cv_min"], invalid_disp)

        # The cost volume has to be multiplied by -1 to be re-considered as a similarity measure
        if cv.attrs['type_measure'] == "max":
            cv['cost_volume'].data *= -1

        cv['cost_volume'].data[invalid_disp] = np.nan
        cv.attrs['optimization'] = 'sgm'

        # Remove temporary values
        del img_ref_crop, img_sec_crop, p1_mat, p2_mat, invalid_disp

        # Maximal cost of the cost volume after optimization
        cmax = invalid_value - 1
        cv.attrs['cmax'] = cmax

        return cv

    def number_of_paths(self, cv: xr.Dataset, disp_paths: np.ndarray, invalid_disp: np.ndarray) -> xr.Dataset:
        """
        Update the confidence measure by adding the number of disp indicator, which gives the number (between 0 and 8)
        of local disparities equal to the ones which return the global minimal costs

        :param cv: the original cost volume dataset
        :type cv:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :param disp_paths: the disparities given the minimum cost
        :type disp_paths: numpy.array
        :param invalid_disp: invalid pixels of the cost_volume
        :type invalid_disp: np.ndarray
        :return: the cost volume dataset updated with a new indicator
        :rtype:
            xarray.Dataset, with the data variables:
                - cost_volume 3D xarray.DataArray (row, col, disp)
                - confidence_measure 3D xarray.DataArray (row, col, indicator)
        """
        # Compute the disparity given the minimum cost volume for each pixel
        disp_map = cv['cost_volume'].fillna(np.inf).argmin(dim='disp')

        # Invalid values of the disparity map
        invalid_mc = np.sum(invalid_disp, axis=2)
        invalid_pixel = np.where(invalid_mc == len(cv.coords['disp']))
        disp_map[invalid_pixel] = -9999

        # Add a new indicator to the confidence measure DataArray
        row, col, nb_indicator = cv['confidence_measure'].shape
        conf_measure = np.zeros((row, col, nb_indicator+1), dtype=np.float32)
        conf_measure[:, :, :-1] = cv['confidence_measure'].data

        indicator = np.copy(cv.coords['indicator'])
        indicator = np.append(indicator, 'optimization_pluginlibSGM_nbOfDisp')

        # Remove confidence_measure dataArray from the dataset to update it
        cv = cv.drop_dims('indicator')
        cv = cv.assign_coords(indicator=indicator)
        cv['confidence_measure'] = xr.DataArray(data=conf_measure, dims=['row', 'col', 'indicator'])

        # Allocate the number of paths given the same disparity as the one which has calculated the min cost
        for d in range(disp_paths.shape[2]):
            pos_y, pos_x = np.where(disp_paths[:, :, d] == disp_map)
            cv['confidence_measure'].data[pos_y, pos_x, -1] += 1

        return cv
