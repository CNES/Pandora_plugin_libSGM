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

import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union

import numpy as np
import xarray as xr


class AbstractPenalty:
    """
    Penalty abstract class
    """

    __metaclass__ = ABCMeta

    penalty_methods_avail = {}

    def __new__(
        cls, directions: List[List[int]], **cfg: Union[str, int, float, bool]
    ):  # pylint: disable=unused-argument
        """
        Return the plugin associated with the penalty_method given in the configuration

        :param directions: directions to
        :type directions: list of [x offset, y offset]
        :param cfg: configuration {'penalty_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractPenalty:
            if isinstance(cfg["penalty_method"], str):
                try:
                    return super(AbstractPenalty, cls).__new__(cls.penalty_methods_avail[cfg["penalty_method"]])
                except KeyError:
                    logging.error("No penalty method named % supported", cfg["penalty_method"])
                    sys.exit(1)
            else:
                if isinstance(cfg["penalty_method"], unicode):  # type: ignore # pylint: disable=undefined-variable
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractPenalty, cls).__new__(
                            cls.penalty_methods_avail[cfg["penalty_method"].encode("utf-8")]
                        )
                    except KeyError:
                        logging.error("No penalty method named % supported", cfg["penalty_method"])
                        sys.exit(1)
        else:
            return super(AbstractPenalty, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name: str, *args: str):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        :param args: allows to register one plugin that contains different methods
        :param args: string
        """

        def decorator(subclass):
            """
            Registers the subclass in the available methods

            :param subclass: the subclass to be registered
            :type subclass: object
            """
            cls.penalty_methods_avail[short_name] = subclass
            for arg in args:
                cls.penalty_methods_avail[arg] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the penalty method

        """
        print("Penalty method description")

    @abstractmethod
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
        :return: invalid values, P1 and P2 penalties
        :rtype: tuple(float, numpy array, numpy array)
        """

    @staticmethod
    def compute_gradient(img: np.ndarray, direction: List[int]) -> np.ndarray:
        """
        Compute image gradient

        :param img: image
        :type img: numpy array of shape(n,m)
        :param direction: directions to
        :type direction: list of [x offset, y offset]
        :return: Gradient
        :rtype: numpy array of shape(n-dir[0], m-dir[1])
        """
        mat1 = img[
            max(direction[0], 0) : min(img.shape[0] + direction[0], img.shape[0]),
            max(direction[1], 0) : min(img.shape[1] + direction[1], img.shape[1]),
        ]
        mat2 = img[
            max(-direction[0], 0) : min(img.shape[0] - direction[0], img.shape[0]),
            max(-direction[1], 0) : min(img.shape[1] - direction[1], img.shape[1]),
        ]

        return np.abs(mat1 - mat2)
