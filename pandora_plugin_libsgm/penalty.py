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
This module provides class and functions to compute penalties used to optimize the cost volume using the LibSGM library
"""

import sys
import logging
from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np


class AbstractPenalty(object):
    __metaclass__ = ABCMeta

    penalty_methods_avail = {}

    def __new__(cls, directions, **cfg):
        """
        Return the plugin associated with the penality_method given in the configuration

        :param directions: directions used in SGM
        :type directions: list of lists
        :param cfg: configuration {'penalty_method': value}
        :type cfg: dictionary
        """
        if cls is AbstractPenalty:
            if type(cfg['penalty_method']) is str:
                try:
                    return super(AbstractPenalty, cls).__new__(cls.penalty_methods_avail[cfg['penalty_method']])
                except KeyError:
                    logging.error("No penalty method named {} supported".format(cfg['penalty_method']))
                    sys.exit(1)
            else:
                if type(cfg['penalty_method']) is unicode:
                    # creating a plugin from registered short name given as unicode (py2 & 3 compatibility)
                    try:
                        return super(AbstractPenalty, cls).__new__(cls.penality_methods_avail[cfg['penalty_method'].encode('utf-8')])
                    except KeyError:
                        logging.error("No penalty method named {} supported".format(cfg['penalty_method']))
                        sys.exit(1)
        else:
            return super(AbstractPenalty, cls).__new__(cls)

    @classmethod
    def register_subclass(cls, short_name):
        """
        Allows to register the subclass with its short name

        :param short_name: the subclass to be registered
        :type short_name: string
        """
        def decorator(subclass):
            cls.penalty_methods_avail[short_name] = subclass
            return subclass

        return decorator

    @abstractmethod
    def desc(self):
        """
        Describes the penalty method

        """
        print('Penalty method description')

    @abstractmethod
    def compute_penalty(self, cv, img_ref, img_sec) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute penalty

        :param cv: the cost volume
        :type cv: xarray.Dataset, with the data variables cost_volume 3D xarray.DataArray (row, col, disp)
        :param img_ref: reference  image
        :type img_ref: numpy array
        :param img_sec: secondary image
        :type img_sec: numpy array
        :return: P1 and P2 penalities
        :rtype: tuple(numpy array, numpy array)
        """
