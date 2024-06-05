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
This module provides class and functions to compute penalties used to optimize the cost volume using the LibSGM library
"""

from typing import Dict, Union, Tuple, List

import numpy as np
import xarray as xr
from json_checker import Checker, And, Or
from pandora.common import is_method

from pandora_plugin_libsgm.penalty import penalty


@penalty.AbstractPenalty.register_subclass("sgm_penalty")
class SgmPenalty(penalty.AbstractPenalty):
    """

    SGM Penalty

    """

    # Default configuration, do not change these values
    _P1 = 8
    _P2 = 32
    _ALPHA = 1.0
    _BETA = 1
    _GAMMA = 1
    _P2_METHOD = "constant"

    def __init__(self, directions: List[List[int]], **cfg: Union[str, int, float, bool]):
        """
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :param cfg: optional configuration, {'P1': value, 'P2': value, 'alpha': value, 'beta': value, 'gamma": value,
                                            'p2_method': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._p1 = self.cfg["P1"]
        self._p2 = self.cfg["P2"]
        self._alpha = self.cfg["alpha"]
        self._beta = self.cfg["beta"]
        self._gamma = self.cfg["gamma"]
        self._p2_method = self.cfg["p2_method"]
        self._directions = directions

    def check_conf(self, **cfg: Union[str, int, float, bool]) -> Dict[str, Union[str, int, float, bool]]:
        """
        Add default values to the dictionary if there are missing elements and check if the dictionary is correct

        :param cfg: optimization configuration
        :type cfg: dict
        :return cfg: optimization configuration updated
        :rtype cfg: dict
        """
        # Give the default value if the required element is not in the configuration
        if "P1" not in cfg:
            cfg["P1"] = self._P1
        if "P2" not in cfg:
            cfg["P2"] = self._P2
        if "alpha" not in cfg:
            cfg["alpha"] = self._ALPHA
        if "beta" not in cfg:
            cfg["beta"] = self._BETA
        if "gamma" not in cfg:
            cfg["gamma"] = self._GAMMA
        if "p2_method" not in cfg:
            cfg["p2_method"] = self._P2_METHOD

        p1_value = cfg["P1"]

        schema = {
            "penalty_method": And(str, lambda x: is_method(x, ["sgm_penalty"])),
            "P1": And(Or(int, float), lambda x: x > 0),
            "P2": And(Or(int, float), lambda x: x > p1_value),
            "alpha": And(Or(int, float), lambda x: x >= 0),
            "beta": And(Or(int, float), lambda x: x > 0),
            "gamma": And(Or(int, float), lambda x: x > 0),
            "p2_method": And(str, lambda x: is_method(x, ["constant", "negativeGradient", "inverseGradient"])),
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the penalty method

        """
        print("SGM penalty method description")

    def compute_penalty(
        self, cv: xr.Dataset, img_left_array: np.ndarray, img_right_array: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute penalty

        :param cv: the cost volume, with the data variables:

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left_array: left  image
        :type img_left_array: np.ndarray
        :param img_right_array: right  image
        :type img_right_array: np.ndarray
        :return: invalid disp, P1 and P2 penalties
        :rtype: tuple(float, numpy array, numpy array)
        """

        # Calculation of the invalid value according to the chosen P2 estimation method
        invalid_value = None
        if self._p2_method == "constant":
            invalid_value = float(cv.attrs["cmax"] + self._p2 + 1)
        elif self._p2_method == "negativeGradient":
            invalid_value = float(cv.attrs["cmax"] + self._gamma + 1)
        elif self._p2_method == "inverseGradient":
            invalid_value = float(cv.attrs["cmax"] + self._gamma + (self._alpha / self._beta) + 1)  # type: ignore

        # Compute penalties
        if self._p2_method == "negativeGradient":
            p1_mask, p2_mask = self.negative_penalty_function(
                img_left_array,
                self._p1,  # type: ignore
                self._p2,  # type: ignore
                self._directions,
                self._alpha,  # type: ignore
                self._gamma,  # type: ignore
            )

        elif self._p2_method == "inverseGradient":
            p1_mask, p2_mask = self.inverse_penalty_function(
                img_left_array,
                self._p1,  # type: ignore
                self._p2,  # type: ignore
                self._directions,
                self._alpha,  # type: ignore
                self._beta,  # type: ignore
                self._gamma,  # type: ignore
            )

        else:
            # Default p2_method is constant
            p1_mask, p2_mask = self.constant_penalty_function(
                img_left_array, self._p1, self._p2, self._directions  # type: ignore
            )

        return invalid_value, p1_mask, p2_mask

    def negative_penalty_function(
        self,
        img_left: np.ndarray,
        p1: Union[int, float],
        p2: Union[int, float],
        directions: List[List[int]],
        alpha: float,
        gamma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute negative penalty

        :param img_left: left  image
        :type img_left: numpy array
        :param p1: default P1 penalty
        :type p1: int or float
        :param p2: default P2 penalty
        :type p2: int or float
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :param alpha: hyper parameter
        :type alpha: float
        :param gamma: hyper parameter
        :type gamma: float
        :return: P1 and P2 penalties
        :rtype: tuple(numpy array, numpy array)
        """
        p1_mask = p1 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)
        p2_mask = p2 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)

        nb_directions = len(directions)
        for i in range(nb_directions):
            direction = directions[i]
            abs_gradient = self.compute_gradient(img_left[:, :], direction)
            val = -alpha * abs_gradient + gamma
            p2_mask[
                max(0, direction[0]) : min(img_left.shape[0] + direction[0], img_left.shape[0]),
                max(0, direction[1]) : min(img_left.shape[1] + direction[1], img_left.shape[1]),
                i,
            ] = val
        # if p2 < defaultP2 then p2
        msk = p2_mask < p2
        p2_mask = p2 * msk + p2_mask * (1 - msk)
        return p1_mask, p2_mask

    def inverse_penalty_function(
        self,
        img_left: np.ndarray,
        p1: Union[int, float],
        p2: Union[int, float],
        directions: List[List[int]],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute inverse penalty

        :param img_left: left  image
        :type img_left: numpy array
        :param p1: P1 penalty
        :type p1: int or float
        :param p2: P2 penalty
        :type p2: int or float
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :param alpha: hyper parameter
        :type alpha: float
        :param beta: hyper parameter
        :type beta: float
        :param gamma: hyper parameter
        :type gamma: float
        :return: P1 and P2 penalties
        :rtype: tuple(numpy array, numpy array)
        """
        p1_mask = p1 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)
        p2_mask = p2 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)

        nb_directions = len(directions)
        for i in range(nb_directions):
            direc = directions[i]
            abs_gradient = self.compute_gradient(img_left[:, :], direc)
            val = alpha / (abs_gradient + beta) + gamma
            p2_mask[
                max(0, direc[0]) : min(img_left.shape[0] + direc[0], img_left.shape[0]),
                max(0, direc[1]) : min(img_left.shape[1] + direc[1], img_left.shape[1]),
                i,
            ] = val
        # if p2 < defaultP2 then p2
        msk = p2_mask < p2
        p2_mask = p2 * msk + p2_mask * (1 - msk)
        return p1_mask, p2_mask

    @staticmethod
    def constant_penalty_function(
        img_left: np.ndarray, p1: Union[int, float], p2: Union[int, float], directions: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute constant penalty

        :param img_left: left  image
        :type img_left: numpy array
        :param p1: P1 penalty
        :type p1: int or float
        :param p2: P2 penalty
        :type p2: int or float
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :return: P1 and P2 penalties
        :rtype: tuple(numpy array, numpy array)
        """
        p1_mask = p1 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)
        p2_mask = p2 * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)
        return p1_mask, p2_mask
