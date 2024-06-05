#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
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
This module provides functions to test sgm penalties
"""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from pandora_plugin_libsgm.penalty import penalty_sgm


@pytest.fixture()
def segment_penalty():
    """Segment penalty fixture."""
    cfg = {
        "P1": 8,
        "P2": 10,
        "alpha": 1.0,
        "gamma": 1,
        "beta": 1,
        "p2_method": "constant",
        "penalty_method": "sgm_penalty",
    }

    directions = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
        [0, -1],
        [-1, 0],
        [-1, -1],
        [-1, 1],
    ]

    return penalty_sgm.SgmPenalty(directions, **cfg)


class TestPenalitySGM:
    """
    TestPenalitySGM class allows to test penality_sgm
    """

    def test_gradient(self, segment_penalty):
        """
        Test Computation of gradient

        """

        img_left = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TEST 1
        img_wanted = np.array([[3, 3, 3], [3, 3, 3]])

        direction = [1, 0]
        computed_gradient = segment_penalty.compute_gradient(img_left, direction)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 2
        img_wanted = np.array([[1, 1], [1, 1], [1, 1]])

        direction = [0, 1]
        computed_gradient = segment_penalty.compute_gradient(img_left, direction)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 3
        img_wanted = np.array([[4, 4], [4, 4]])

        direction = [1, 1]
        computed_gradient = segment_penalty.compute_gradient(img_left, direction)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 4
        img_wanted = np.array([[4, 4], [4, 4]])

        direction = [-1, -1]
        computed_gradient = segment_penalty.compute_gradient(img_left, direction)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 5
        img_wanted = np.array([[2, 2], [2, 2]])

        direction = [-1, 1]
        computed_gradient = segment_penalty.compute_gradient(img_left, direction)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

    def test_constant_penalty_function(self, segment_penalty):
        """
        Test Computation of gradient

        """
        default_p1 = 1
        default_p2 = 2

        img_left = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TEST 1
        p1_wanted = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        )

        p2_wanted = np.array(
            [
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            ]
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        computed_p1, computed_p2 = segment_penalty.constant_penalty_function(
            img_left, default_p1, default_p2, directions
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1, p1_wanted)
        np.testing.assert_array_equal(computed_p2, p2_wanted)

    def test_inverse_penalty_function(self, segment_penalty):
        """
        Test Computation of gradient

        """
        default_p1 = 1
        default_p2 = 2

        img_left = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TEST 1
        p1_wanted = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        )

        p2_wanted = np.array(
            [
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            ]
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        alpha = 1
        beta = 1
        gamma = 1
        computed_p1, computed_p2 = segment_penalty.inverse_penalty_function(
            img_left, default_p1, default_p2, directions, alpha, beta, gamma
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1, p1_wanted)
        np.testing.assert_array_equal(computed_p2, p2_wanted)

        # TEST 2

        default_p1 = 1
        default_p2 = 0

        img_left = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
        )
        p1_wanted_0 = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        p2_wanted_0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 6],
                [7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 6],
                [7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 6],
                [7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 6],
            ],
            dtype=np.float32,
        )

        p2_wanted_1 = np.array(
            [
                [0, 6 / 5, 6 / 5, 6 / 5, 6 / 5],
                [0, 6 / 5, 6 / 5, 6 / 5, 6 / 5],
                [0, 6 / 5, 6 / 5, 6 / 5, 6 / 5],
                [0, 6 / 5, 6 / 5, 6 / 5, 6 / 5],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        p2_wanted_2 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 8 / 7, 8 / 7, 8 / 7, 8 / 7],
                [0, 8 / 7, 8 / 7, 8 / 7, 8 / 7],
                [0, 8 / 7, 8 / 7, 8 / 7, 8 / 7],
                [0, 8 / 7, 8 / 7, 8 / 7, 8 / 7],
            ],
            dtype=np.float32,
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        alpha = 1
        beta = 1
        gamma = 1
        computed_p1, computed_p2 = segment_penalty.inverse_penalty_function(
            img_left, default_p1, default_p2, directions, alpha, beta, gamma
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1[:, :, 0], p1_wanted_0)
        np.testing.assert_array_equal(computed_p2[:, :, 0], p2_wanted_0)
        np.testing.assert_array_equal(computed_p2[:, :, 1], p2_wanted_1)
        np.testing.assert_array_equal(computed_p2[:, :, 2], p2_wanted_2)

    def test_negative_penalty_function(self, segment_penalty):
        """
        Test Computation of gradient

        """
        default_p1 = 1
        default_p2 = 2

        img_left = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TEST 1
        p1_wanted = np.array(
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ]
        )

        p2_wanted = np.array(
            [
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            ]
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        alpha = 1
        gamma = 1
        computed_p1, computed_p2 = segment_penalty.negative_penalty_function(
            img_left, default_p1, default_p2, directions, alpha, gamma
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1, p1_wanted)
        np.testing.assert_array_equal(computed_p2, p2_wanted)

        # TEST 2

        default_p1 = 1
        default_p2 = 0

        img_left = np.array(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
        )
        p1_wanted_0 = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

        p2_wanted_0 = np.array(
            [
                [0, 0, 0, 0, 0],
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
                [5, 5, 5, 5, 5],
            ]
        )

        p2_wanted_1 = np.array(
            [
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
                [0, 6, 6, 6, 6],
                [0, 0, 0, 0, 0],
            ]
        )

        p2_wanted_2 = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 4, 4, 4, 4],
                [0, 4, 4, 4, 4],
                [0, 4, 4, 4, 4],
                [0, 4, 4, 4, 4],
            ]
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        alpha = 1
        gamma = 10
        computed_p1, computed_p2 = segment_penalty.negative_penalty_function(
            img_left, default_p1, default_p2, directions, alpha, gamma
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1[:, :, 0], p1_wanted_0)
        np.testing.assert_array_equal(computed_p2[:, :, 0], p2_wanted_0)
        np.testing.assert_array_equal(computed_p2[:, :, 1], p2_wanted_1)
        np.testing.assert_array_equal(computed_p2[:, :, 2], p2_wanted_2)
