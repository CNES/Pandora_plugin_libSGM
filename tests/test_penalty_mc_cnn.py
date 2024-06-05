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
This module provides functions to test mc-cnn penalties
"""

# pylint: disable=redefined-outer-name
import numpy as np
import pytest

from pandora_plugin_libsgm.penalty import penalty_mc_cnn


@pytest.fixture()
def mccnn_penalty():
    """Mc Cnn penalty fixture."""
    cfg = {
        "P1": 8,
        "P2": 10,
        "Q1": 1.0,
        "Q2": 1.0,
        "D": 1.0,
        "V": 1.0,
        "penalty_method": "mc_cnn_fast_penalty",
    }

    directions = [[0, 1], [1, 0], [1, 1], [1, -1], [0, -1], [-1, 0], [-1, -1], [-1, 1]]

    return penalty_mc_cnn.MccnnPenalty(directions, **cfg)


class TestPenalitySGM:
    """
    TestPenalitySGM class allows to test penality_sgm
    """

    def test_gradient(self, mccnn_penalty):
        """
        Test Computation of gradient

        """

        img_left = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # TEST 1
        img_wanted = np.array([[3, 3, 3], [3, 3, 3]])

        direc = [1, 0]
        computed_gradient = mccnn_penalty.compute_gradient(img_left, direc)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 2
        img_wanted = np.array([[1, 1], [1, 1], [1, 1]])

        direc = [0, 1]
        computed_gradient = mccnn_penalty.compute_gradient(img_left, direc)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 3
        img_wanted = np.array([[4, 4], [4, 4]])

        direc = [1, 1]
        computed_gradient = mccnn_penalty.compute_gradient(img_left, direc)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 4
        img_wanted = np.array([[4, 4], [4, 4]])

        direc = [-1, -1]
        computed_gradient = mccnn_penalty.compute_gradient(img_left, direc)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

        # TEST 5
        img_wanted = np.array([[2, 2], [2, 2]])

        direc = [-1, 1]
        computed_gradient = mccnn_penalty.compute_gradient(img_left, direc)
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_gradient, img_wanted)

    def test_mc_cnn_penalty_function(self, mccnn_penalty):
        """
        Test mc_cnn menalty

        """

        # TEST 1

        default_p1 = 3
        default_p2 = 4
        d_mccnn = 5
        v_mccnn = 2
        q1_mccnn = 5
        q2_mccnn = 6

        img_right = np.array(
            [
                [1, 6, 11, 16, 21],
                [2, 7, 12, 17, 22],
                [3, 8, 13, 18, 23],
                [4, 9, 14, 19, 24],
                [5, 10, 15, 20, 25],
            ]
        )

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
                [3, 3, 3, 3, 3],
                [3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5],
                [3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5],
                [3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5],
                [3 / 5, 3 / 5, 3 / 5, 3 / 5, 3 / 5],
            ],
            dtype=np.float32,
        )

        p2_wanted_0 = np.array(
            [
                [4, 4, 4, 4, 4],
                [4 / 5, 4 / 5, 4 / 5, 4 / 5, 4 / 5],
                [4 / 5, 4 / 5, 4 / 5, 4 / 5, 4 / 5],
                [4 / 5, 4 / 5, 4 / 5, 4 / 5, 4 / 5],
                [4 / 5, 4 / 5, 4 / 5, 4 / 5, 4 / 5],
            ],
            dtype=np.float32,
        )

        # divide by V on direction in [1,5]
        p1_wanted_1 = np.array(
            [
                [3 / 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2],
                [3 / 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2],
                [3 / 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2],
                [3 / 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2],
                [3 / 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2],
            ],
            dtype=np.float32,
        )

        p2_wanted_1 = np.array(
            [
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
            ]
        )

        p1_wanted_2 = np.array(
            [
                [3, 3, 3, 3, 3],
                [3, 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6)],
                [3, 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6)],
                [3, 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6)],
                [3, 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6), 3 / (5 * 6)],
            ],
            dtype=np.float32,
        )

        p2_wanted_2 = np.array(
            [
                [4, 4, 4, 4, 4],
                [4, 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6)],
                [4, 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6)],
                [4, 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6)],
                [4, 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6), 4 / (5 * 6)],
            ],
            dtype=np.float32,
        )

        directions = [[1, 0], [-1, 1], [1, 1]]
        computed_p1, computed_p2 = mccnn_penalty.mc_cnn_penalty_function(
            img_left,
            img_right,
            default_p1,
            default_p2,
            q1_mccnn,
            q2_mccnn,
            d_mccnn,
            v_mccnn,
            directions,
        )
        # Check if the calculated gradient is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(computed_p1[:, :, 0], p1_wanted_0)
        np.testing.assert_array_equal(computed_p2[:, :, 0], p2_wanted_0)
        np.testing.assert_array_equal(computed_p1[:, :, 1], p1_wanted_1)
        np.testing.assert_array_equal(computed_p2[:, :, 1], p2_wanted_1)
        np.testing.assert_array_equal(computed_p1[:, :, 2], p1_wanted_2)
        np.testing.assert_array_equal(computed_p2[:, :, 2], p2_wanted_2)

    # pylint:disable=protected-access
    @staticmethod
    def test_penalty_default_configuration():
        """
        Test default parameters according to the type of mc-cnn measure

        """
        # Check with fast mc-cnn
        cfg = {
            "penalty_method": "mc_cnn_fast_penalty",
        }

        _directions = [
            [0, 1],
            [1, 0],
            [1, 1],
            [1, -1],
            [0, -1],
            [-1, 0],
            [-1, -1],
            [-1, 1],
        ]
        penalty = penalty_mc_cnn.MccnnPenalty(_directions, **cfg)

        # Checks that penalties correspond to the type of measure
        assert penalty._p1 == penalty_mc_cnn.MccnnPenalty._P1
        assert penalty._p2 == penalty_mc_cnn.MccnnPenalty._P2
        assert penalty._q1 == penalty_mc_cnn.MccnnPenalty._Q1
        assert penalty._q2 == penalty_mc_cnn.MccnnPenalty._Q2
        assert penalty._d == penalty_mc_cnn.MccnnPenalty._D
        assert penalty._v == penalty_mc_cnn.MccnnPenalty._V
