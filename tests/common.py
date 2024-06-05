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
This module contains common functions present in Plugin_libsgm's tests.
"""

import numpy as np


def error_mask(data: np.ndarray, gt: np.ndarray) -> float:
    """
    Percentage of bad pixels ( != ground truth ) in the validity mask
    :param data: generated array in test
    :type data: np.ndarray
    :param gt: Ground truth
    :type gt: np.ndarray
    :return: Percentage
    :rtype: float
    """

    nb_rows, nb_cols = data.shape
    nb_error = 0
    for row in range(nb_rows):
        for col in range(nb_cols):
            if data[row, col] != gt[row, col]:
                nb_error += 1

    return nb_error / float(nb_rows * nb_cols)


def strict_error(data: np.ndarray, gt: np.ndarray) -> float:
    """
    Average of bad pixels  ( != ground truth )
    :param data: generated array in test
    :type data: np.ndarray
    :param gt: Ground truth
    :type gt: np.ndarray
    :return: Percentage
    :rtype: float
    """

    nb_rows, nb_cols = data.shape
    nb_error = 0
    for row in range(nb_rows):
        for col in range(nb_cols):
            if data[row, col] != gt[row, col]:
                nb_error += 1

    return nb_error / float(nb_rows * nb_cols)


def error(
    data: np.ndarray, gt: np.ndarray, threshold: int, unknown_disparity: int = 0, flag_inverse_value: bool = True
) -> float:
    """
    Percentage of bad pixels whose error is > threshold
    If the GT is tested : the values for the left image might be inverse

    :param data: generated array in test
    :type data: np.ndarray
    :param gt: Ground truth
    :type gt: np.ndarray
    :param threshold: accepted threshold in test
    :type threshold: int
    :param unknown_disparity: value for invalid disp in GT
    :type unknown_disparity: int
    :param flag_inverse_value: Whether the GT and disparity map values are reversed
    :type flag_inverse_value: bool
    :return: Percentage
    :rtype: float
    """

    if flag_inverse_value:
        data = -1 * data
    nb_row, nb_col = data.shape
    nb_error = 0
    for row in range(nb_row):
        for col in range(nb_col):
            if gt[row, col] != unknown_disparity:
                if abs((data[row, col] + gt[row, col])) > threshold:
                    nb_error += 1

    return nb_error / float(nb_row * nb_col)
