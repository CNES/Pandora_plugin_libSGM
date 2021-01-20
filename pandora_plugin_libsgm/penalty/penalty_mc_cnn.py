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
This module provides class and functions to compute penalties used to optimize the cost volume using the LibSGM library
"""

from typing import Dict, Union, Tuple, List

import numpy as np
import xarray as xr
from json_checker import Checker, And, Or

from pandora.common import is_method
from pandora_plugin_libsgm.penalty import penalty


@penalty.AbstractPenalty.register_subclass('mc_cnn_penalty')
class MccnnPenalty(penalty.AbstractPenalty):
    """

    MC-CNN Penalty

    """

    _P1 = 1.3
    _P2 = 18.1
    _Q1 = 4.5
    _Q2 = 9
    _D = 0.13
    _V = 2.75
    _OVERCOUNTING = False
    _MIN_COST_PATH = False

    def __init__(self, directions: List[List[int]], **cfg: Union[str, int, float, bool]):
        """
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :param cfg: optional configuration, {'P1': value, 'P2': value, 'Q1': value, 'Q2': value, 'D": value,
                                            'V': value}
        :type cfg: dict
        """
        self.cfg = self.check_conf(**cfg)
        self._p1 = self.cfg['P1']
        self._p2 = self.cfg['P2']
        self._q1 = self.cfg['Q1']
        self._q2 = self.cfg['Q2']
        self._d = self.cfg['D']
        self._v = self.cfg['V']
        self._overcounting = self.cfg['overcounting']
        self._min_cost_paths = self.cfg['min_cost_paths']
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
        if 'P1' not in cfg:
            cfg['P1'] = self._P1
        if 'P2' not in cfg:
            cfg['P2'] = self._P2
        if 'Q1' not in cfg:
            cfg['Q1'] = self._Q1
        if 'Q2' not in cfg:
            cfg['Q2'] = self._Q2
        if 'D' not in cfg:
            cfg['D'] = self._D
        if 'V' not in cfg:
            cfg['V'] = self._V
        if 'overcounting' not in cfg:
            cfg['overcounting'] = self._OVERCOUNTING
        if 'min_cost_paths' not in cfg:
            cfg['min_cost_paths'] = self._MIN_COST_PATH

        p1_value = cfg['P1']
        schema = {
            'sgm_version': And(str, lambda x: is_method(x, ['c++', 'python_libsgm', 'python_libsgm_parall'])),
            'optimization_method': And(str, lambda x: is_method(x, ['sgm'])),
            'penalty_method': And(str, lambda x: is_method(x, ['mc_cnn_penalty'])),
            'P1': And(Or(int, float), lambda x: x > 0),
            'P2': And(Or(int, float), lambda x: x > p1_value),
            'Q1': And(Or(int, float), lambda x: x > 0),
            'Q2': And(Or(int, float), lambda x: x > 0),
            'D': And(Or(int, float), lambda x: x >= 0),
            'V': And(Or(int, float), lambda x: x > 0),
            'overcounting': bool,
            'min_cost_paths': bool
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def desc(self):
        """
        Describes the penality method

        """
        print('Penalty method description')

    def compute_penalty(self, cv: xr.Dataset, img_left: np.ndarray, img_right: np.ndarray) \
            -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute penalty

        :param cv: the cost volume, with the data variables:

            - cost_volume 3D xarray.DataArray (row, col, disp)
            - confidence_measure 3D xarray.DataArray (row, col, indicator)
        :type cv: xarray.Dataset
        :param img_left: left  image
        :type img_left: numpy array
        :param img_right: right  image
        :type img_right: numpy array
        :return: P1 and P2 penalities
        :rtype: tuple(numpy array, numpy array)
        """

        # Calculation of the invalid value
        p2_max = max(self._p2, self._p2 / self._q2, self._p2 / self._p1)
        invalid_value = float(cv.attrs['cmax'] + p2_max + 1)

        # Compute penalties
        p1_mask, p2_mask = self.mc_cnn_penalty_function(img_left, img_right, self._p1, self._p2, self._q1, self._q2,
                                                        self._d, self._v, self._directions)

        return invalid_value, p1_mask, p2_mask

    @staticmethod
    def compute_gradient(img: np.ndarray, direction: List[List[int]]) -> np.ndarray:
        """
        Compute image gradient

        :param img: image
        :type img: numpy array of shape(n,m)
        :param direction: directions to
        :type direction: list of [x offset, y offset]
        :return: Gradient
        :rtype: numpy array of shape(n-dir[0], m-dir[1])
        """
        mat1 = img[max(direction[0], 0): min(img.shape[0] + direction[0], img.shape[0]),
               max(direction[1], 0): min(img.shape[1] + direction[1], img.shape[1])]
        mat2 = img[max(-direction[0], 0): min(img.shape[0] - direction[0], img.shape[0]),
               max(-direction[1], 0): min(img.shape[1] - direction[1], img.shape[1])]

        return np.abs(mat1 - mat2)

    def mc_cnn_penalty_function(self, img_left: np.ndarray, img_right: np.ndarray, # pylint: disable=too-many-arguments
                                p1_mccnn: Union[int, float], p2_mccnn: Union[int, float], q1_mccnn: Union[int, float],
                                q2_mccnn: Union[int, float], d_mccnn: Union[int, float], v_mccnn: Union[int, float],
                                directions: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mc_cnn penalty

        :param img_left: left  image
        :type img_left: numpy array
        :param img_right: right  image
        :type img_right: numpy array
        :param p1_mccnn:  P1 penalty
        :type p1_mccnn: int or float
        :param p2_mccnn: default P2 penalty
        :type p2_mccnn: int or float
        :param q1_mccnn: hyper parameter
        :type q1_mccnn: int or float
        :param q2_mccnn: hyper parameter
        :type q2_mccnn: int or float
        :param d_mccnn: hyper parameter
        :type d_mccnn: int or float
        :param v_mccnn: hyper parameter
        :type v_mccnn: int or float
        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :return: P1 and P2 penalties
        :rtype: tuple(numpy array, numpy array)
        """
        p1_mask = p1_mccnn * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)
        p2_mask = p2_mccnn * np.ones([img_left.shape[0], img_left.shape[1], len(directions)], dtype=np.float32)

        nb_directions = len(directions)
        for i in range(nb_directions):
            direction = directions[i]
            abs_gradient_left = self.compute_gradient(img_left[:, :], direction)
            abs_gradient_right = self.compute_gradient(img_right[:, :], direction)
            # if(D1<sgm_D && D2<sgm_D)
            msk1 = (abs_gradient_left < d_mccnn) * (abs_gradient_right < d_mccnn)
            final_p1 = msk1 * p1_mccnn * np.ones(abs_gradient_left.shape)
            final_p2 = msk1 * p2_mccnn * np.ones(abs_gradient_left.shape)
            # if(D1 > sgm_D && D2 > sgm_D)
            msk2 = (abs_gradient_left > d_mccnn) * (abs_gradient_right > d_mccnn)
            final_p1 = final_p1 + msk2 * (p1_mccnn / (q1_mccnn * q2_mccnn)) * np.ones(abs_gradient_left.shape)
            final_p2 = final_p2 + msk2 * (p2_mccnn / (q1_mccnn * q2_mccnn)) * np.ones(abs_gradient_left.shape)
            # else
            msk3 = (1 - msk1) * (1 - msk2)
            final_p1 = final_p1 + msk3 * (p1_mccnn / q1_mccnn) * np.ones(abs_gradient_left.shape)
            final_p2 = final_p2 + msk3 * (p2_mccnn / q1_mccnn) * np.ones(abs_gradient_left.shape)

            p1_mask[max(0, direction[0]): min(img_left.shape[0] + direction[0], img_left.shape[0]),
            max(0, direction[1]): min(img_left.shape[1] + direction[1], img_left.shape[1]), i] = final_p1
            p2_mask[max(0, direction[0]): min(img_left.shape[0] + direction[0], img_left.shape[0]),
            max(0, direction[1]): min(img_left.shape[1] + direction[1], img_left.shape[1]), i] = final_p2

            if i in [1, 5]:
                p1_mask[:, :, i] = p1_mask[:, :, i] / v_mccnn

        return p1_mask, p2_mask
