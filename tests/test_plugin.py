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
This module provides functions to test Pandora + plugin_LibSGM
"""

import unittest

import numpy as np
import rasterio
import xarray as xr

import pandora
from pandora import matching_cost, optimization, cost_volume_confidence
from pandora.state_machine import PandoraMachine
import tests.common as common


class TestPlugin(unittest.TestCase):
    """
    TestPlugin class allows to test pandora + plugin_libsgm
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        self.left = pandora.read_img("tests/left.png", no_data=np.nan, mask=None)
        self.right = pandora.read_img("tests/right.png", no_data=np.nan, mask=None)
        self.disp_left = rasterio.open("tests/disp_left.tif").read(1)
        self.disp_right = rasterio.open("tests/disp_right.tif").read(1)
        self.occlusion = rasterio.open("tests/occl.png").read(1)

        self.disp_left_zncc = rasterio.open("tests/disp_left_zncc.tif").read(1)
        self.disp_right_zncc = rasterio.open("tests/disp_right_zncc.tif").read(1)

    def test_libsgm(self):
        """
        Test pandora + plugin_libsgm

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg["pipeline"])

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if common.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_negative_disparities(self):
        """
        Test pandora + plugin_libsgm, with negative disparities

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, -1, user_cfg["pipeline"])

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if common.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_positive_disparities(self):
        """
        Test pandora + plugin_libsgm, with positive disparities

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        right, left = pandora.run(pandora_machine, self.right, self.left, 1, 60, user_cfg["pipeline"])

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(-1 * right["disparity_map"].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_zncc(self):
        """
        Test pandora + plugin_libsgm if ZNCC measure is used
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm_zncc.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg["pipeline"])

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        if common.strict_error(left["disparity_map"].data[61:-61, 61:-61], self.disp_left_zncc[61:-61, 61:-61]) > 0:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        if common.strict_error(right["disparity_map"].data, self.disp_right_zncc) > 0:
            raise AssertionError

    @staticmethod
    def test_number_of_disp():
        """
        Test plugin_libsgm number_of_disp function if min_cost_paths is activated
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")
        user_cfg["pipeline"]["matching_cost"]["window_size"] = 3
        user_cfg["pipeline"]["optimization"]["min_cost_paths"] = True

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Import pandora plugins
        pandora.import_plugin()

        data = np.array(([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        data = np.array(([1, 1, 1, 2, 2], [1, 1, 1, 4, 2], [1, 1, 1, 4, 4], [1, 1, 1, 1, 1]), dtype=np.float32)
        right = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=-2, disp_max=2)

        # Disparities which give a minimum local cost, in indices
        disp_path = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 2, 3, 1, 3, 3, 2],
                    [0, 1, 1, 4, 2, 2, 3, 1],
                    [2, 4, 2, 4, 3, 0, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 2, 3, 1, 0, 1, 3],
                    [2, 1, 1, 3, 1, 3, 1, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.float32,
        )

        invalid_disp = np.isnan(cv["cost_volume"].data)
        cv_updated = optimization_.number_of_disp(cv, disp_path, invalid_disp)

        # Ground truth calculated with disp_path
        gt_disp = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 2, 3, 0, np.nan],
                [np.nan, 1, 4, 8, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["confidence_measure"].data[:, :, -1], gt_disp)

    @staticmethod
    def test_number_of_disp_with_previous_confidence():
        """
        Test plugin_libsgm number_of_disp function if min_cost_paths is activated and the confidence measure was present
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")
        user_cfg["pipeline"]["matching_cost"]["window_size"] = 3
        user_cfg["pipeline"]["optimization"]["min_cost_paths"] = True

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])
        confidence_ = cost_volume_confidence.AbstractCostVolumeConfidence(
            **user_cfg["pipeline"]["cost_volume_confidence"]
        )

        # Import pandora plugins
        pandora.import_plugin()

        data = np.array(([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        data = np.array(([1, 1, 1, 2, 2], [1, 1, 1, 4, 2], [1, 1, 1, 4, 4], [1, 1, 1, 1, 1]), dtype=np.float32)
        right = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(img_left=left, img_right=right, disp_min=-2, disp_max=2)
        left_disp, cv = confidence_.confidence_prediction(None, left, right, cv)  # pylint:disable=unused-variable
        # Disparities which give a minimum local cost, in indices
        disp_path = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 2, 3, 1, 3, 3, 2],
                    [0, 1, 1, 4, 2, 2, 3, 1],
                    [2, 4, 2, 4, 3, 0, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 2, 3, 1, 0, 1, 3],
                    [2, 1, 1, 3, 1, 3, 1, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.float32,
        )

        invalid_disp = np.isnan(cv["cost_volume"].data)
        cv_updated = optimization_.number_of_disp(cv, disp_path, invalid_disp)

        # Ground truth calculated with disp_path
        gt_disp = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 2, 3, 0, np.nan],
                [np.nan, 1, 4, 8, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["confidence_measure"].data[:, :, -1], gt_disp)

    def test_apply_confidence_no_confidence(self):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Test
        use_confidence = False

        data_cv = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )
        cv_in = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], data_cv)},
            coords={
                "row": np.arange(data_cv.shape[0]),
                "col": np.arange(data_cv.shape[1]),
                "disp": np.arange(data_cv.shape[2]),
            },
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cv_in, use_confidence)

        # Ground Truth
        optim_cv_gt = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["cost_volume"].data[:, :, :], optim_cv_gt)

        # Check if confidence_is_int is right
        self.assertEqual(confidence_is_int, True)

    def test_apply_confidence_no_confidence_dataarray(self):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Test
        use_confidence = True

        data_cv = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )
        cv_in = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], data_cv)},
            coords={
                "row": np.arange(data_cv.shape[0]),
                "col": np.arange(data_cv.shape[1]),
                "disp": np.arange(data_cv.shape[2]),
            },
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cv_in, use_confidence)

        # Ground Truth
        optim_cv_gt = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["cost_volume"].data[:, :, :], optim_cv_gt)

        # Check if confidence_is_int is right
        self.assertEqual(confidence_is_int, True)

    def test_apply_confidence_with_confidence_dataarray(self):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Test

        use_confidence = True

        data_cv = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )

        data_confidence = np.array([[1, 1, 1, 0.5], [1, 1, 0.5, 1], [1, 1, 1, 1]], dtype=np.float32)

        cv_in = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], data_cv)},
            coords={
                "row": np.arange(data_cv.shape[0]),
                "col": np.arange(data_cv.shape[1]),
                "disp": np.arange(data_cv.shape[2]),
            },
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        cv_in["confidence_measure"] = xr.DataArray(data_confidence, dims=["row", "col"])

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cv_in, use_confidence)

        # Ground Truth
        optim_cv_gt = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [0.5, 0.5, 0.5, 0.5, 0.5]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [0.5, 0.5, 0.5, 2, 1.5], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["cost_volume"].data[:, :, :], optim_cv_gt)

        # Check if confidence_is_int is right
        self.assertEqual(confidence_is_int, False)

    @staticmethod
    def test_compute_piecewise_layer_not_in_dataset():
        """
        Test plugin_libsgm compute_piecewise_layer function, with user asking for piecewise optimization,
         without any in dataset
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Create input dataset
        data_img_left = np.random.rand(5, 6)
        img_left = xr.Dataset(
            {"im": (["row", "col"], data_img_left)},
            coords={"row": np.arange(data_img_left.shape[0]), "col": np.arange(data_img_left.shape[1])},
        )

        piecewise_optimization_layer = "toto"

        classif_arr = optimization_.compute_piecewise_layer(img_left, piecewise_optimization_layer)

        gt_classif = np.ones((5, 6))
        np.testing.assert_array_equal(classif_arr, gt_classif)

    @staticmethod
    def test_compute_piecewise_layer_in_dataset():
        """
        Test plugin_libsgm compute_piecewise_layer function, with user asking for piecewise optimization,
         without any in dataset
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Create input dataset
        data_img_left = np.random.rand(3, 2)
        img_left = xr.Dataset(
            {"im": (["row", "col"], data_img_left)},
            coords={"row": np.arange(data_img_left.shape[0]), "col": np.arange(data_img_left.shape[1])},
        )

        data_classif = np.array(([[2, 1], [1, 3], [2, 2.6]]))
        img_left["classif"] = xr.DataArray(data_classif, dims=["row", "col"])

        piecewise_optimization_layer = "classif"

        classif_arr = optimization_.compute_piecewise_layer(img_left, piecewise_optimization_layer)

        gt_classif = np.array(([[2, 1], [1, 3], [2, 2.6]]))
        np.testing.assert_array_equal(classif_arr, gt_classif)

    @staticmethod
    def test_compute_piecewise_none_layer_in_dataset():
        """
        Test plugin_libsgm compute_piecewise_layer function, with user asking for piecewise optimization,
         without any in dataset
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Create input dataset
        data_img_left = np.random.rand(3, 2)
        img_left = xr.Dataset(
            {"im": (["row", "col"], data_img_left)},
            coords={"row": np.arange(data_img_left.shape[0]), "col": np.arange(data_img_left.shape[1])},
        )

        data_classif = np.array(([[2, 1], [1, 3], [2, 2.6]]))
        img_left["classif"] = xr.DataArray(data_classif, dims=["row", "col"])

        piecewise_optimization_layer = "None"

        classif_arr = optimization_.compute_piecewise_layer(img_left, piecewise_optimization_layer)

        gt_classif = np.ones((3, 2))
        np.testing.assert_array_equal(classif_arr, gt_classif)


if __name__ == "__main__":
    unittest.main()
