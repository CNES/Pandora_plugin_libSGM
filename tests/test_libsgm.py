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

import copy
import unittest

import numpy as np
import pandora
import pytest
import rasterio
import xarray as xr
from pandora import matching_cost, optimization, cost_volume_confidence
from pandora.state_machine import PandoraMachine

from tests import common


class TestPluginSGM(unittest.TestCase):
    """
    TestPlugin class allows to test pandora + plugin_libsgm
    """

    def setUp(self):
        """
        Method called to prepare the test fixture

        """

        # Cones images
        self.left_cones = pandora.read_img("tests/inputs/left.png", no_data=np.nan, mask=None)
        self.right_cones = pandora.read_img("tests/inputs/right.png", no_data=np.nan, mask=None)

        # Manually computed images
        data = np.array(([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]), dtype=np.float32)
        self.left_crafted = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": 0,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": None,
                "band_list": None,
            },
        )

        data = np.array(([1, 1, 1, 2, 2], [1, 1, 1, 4, 2], [1, 1, 1, 4, 4], [1, 1, 1, 1, 1]), dtype=np.float32)
        self.right_crafted = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": 0,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": None,
                "band_list": None,
            },
        )

        # Create cost volume
        data_cv = np.array(
            [
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
                [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )
        self.cv = xr.Dataset(
            {"cost_volume": (["row", "col", "disp"], data_cv)},
            coords={
                "row": np.arange(data_cv.shape[0]),
                "col": np.arange(data_cv.shape[1]),
                "disp": np.arange(data_cv.shape[2]),
                "indicator": ["ambiguity_confidence"],
            },
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        # Cones outputs
        self.disp_left = rasterio.open("tests/outputs/disp_left.tif").read(1)
        self.disp_right = rasterio.open("tests/outputs/disp_right.tif").read(1)
        self.occlusion = rasterio.open("tests/outputs/occl.png").read(1)

        self.disp_left_zncc = rasterio.open("tests/outputs/disp_left_zncc.tif").read(1)
        self.disp_right_zncc = rasterio.open("tests/outputs/disp_right_zncc.tif").read(1)

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
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1, flag_inverse_value=False) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2, flag_inverse_value=False) > 0.15:
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
        if common.error(right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(right["disparity_map"].data, self.disp_right, 2) > 0.15:
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
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, -1, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1, flag_inverse_value=False) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2, flag_inverse_value=False) > 0.15:
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
        if common.error(right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(right["disparity_map"].data, self.disp_right, 2) > 0.15:
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

        right, left = pandora.run(pandora_machine, self.right_cones, self.left_cones, 1, 60, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 1, flag_inverse_value=False) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, self.disp_left, 2, flag_inverse_value=False) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(right["disparity_map"].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(right["disparity_map"].data, self.disp_right, 2) > 0.15:
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
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        if common.strict_error(left["disparity_map"].data[61:-61, 61:-61], self.disp_left_zncc[61:-61, 61:-61]) > 0:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        if common.strict_error(right["disparity_map"].data, self.disp_right_zncc) > 0:
            raise AssertionError

    def test_number_of_disp(self):
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

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(
            img_left=self.left_crafted, img_right=self.right_crafted, disp_min=-2, disp_max=2
        )

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

    def test_number_of_disp_with_previous_confidence(self):
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

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(
            img_left=self.left_crafted, img_right=self.right_crafted, disp_min=-2, disp_max=2
        )
        _, cv = confidence_.confidence_prediction(None, self.left_crafted, self.right_crafted, cv)
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

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(self.cv, use_confidence)

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

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(self.cv, use_confidence)

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

        data_confidence = np.expand_dims(
            np.array([[1, 1, 1, 0.5], [1, 1, 0.5, 1], [1, 1, 1, 1]], dtype=np.float32), axis=2
        )

        cv_in = copy.deepcopy(self.cv)
        cv_in["confidence_measure"] = xr.DataArray(data_confidence, dims=["row", "col", "indicator"])

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

    def test_optimization_layer_with_sgm(self):
        """
        Test the optimization layer function with sgm default configuration
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        cv_in = copy.deepcopy(self.cv)

        prior_array_out = optimization_.compute_optimization_layer(
            cv_in, self.left_crafted, self.left_crafted["im"].shape
        )

        # Check that cost volume isn't changed
        with pytest.raises(KeyError):
            _ = cv_in["internal"]

        # Check that prior array is the default one
        default_prior_array = np.ones(self.left_crafted["im"].data.shape, dtype=np.float32)
        np.testing.assert_array_equal(default_prior_array, prior_array_out)

    def test_user_initiate_sgm_with_geomprior(self):
        """
        Test that user can't implement geometric_prior with a sgm configuration
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, -1, user_cfg)

    def test_optimization_layer_with_multiband(self):
        """
        Test the optimization layer function with multiband input images
        """
        # Read input rgb images
        left_rgb = pandora.read_img("tests/inputs/left_rgb.tif", no_data=np.nan, mask=None)
        right_rgb = pandora.read_img("tests/inputs/right_rgb.tif", no_data=np.nan, mask=None)

        # Prepare the configuration for multiband
        user_cfg = pandora.read_config_file("tests/conf/sgm.json")
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"
        # We choose the mccnn penalty to verify that the correct band is being used
        user_cfg["pipeline"]["optimization"]["penalty"]["penalty_method"] = "mc_cnn_fast_penalty"
        user_cfg["pipeline"]["optimization"]["penalty"].pop("p2_method")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        # Import pandora plugins
        pandora.import_plugin()

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(img_left=left_rgb, img_right=right_rgb, disp_min=-60, disp_max=0)
        cv_in = copy.deepcopy(cv)
        # Get invalid disparities of the cost volume
        invalid_disp = np.isnan(cv["cost_volume"].data)

        # Obtain optimized cost volume for a multiband input
        out_cv = optimization_.optimize_cv(cv, left_rgb, right_rgb)
        # Invalid disparities of the cost volume as set as -9999
        out_cv["cost_volume"].data[invalid_disp] = -9999

        # To verify if the correct band is being used, we perform the optimize_cv steps
        # selecting manually the band
        # Image bands are "r","g","b". "g" has been chosen for matching cost, hence the band is 1
        band = 1
        # Get the image band and optimize cv
        img_left_array = np.ascontiguousarray(left_rgb["im"].data[band, :, :], dtype=np.float32)
        img_right_array = np.ascontiguousarray(right_rgb["im"].data[band, :, :], dtype=np.float32)
        invalid_value, p1_mat, p2_mat = optimization_._penalty.compute_penalty(  # pylint:disable=protected-access
            cv_in, img_left_array, img_right_array
        )
        cv_in, confidence_is_int = optimization_.apply_confidence(
            cv_in, optimization_._use_confidence  # pylint:disable=protected-access
        )
        optimization_layer = optimization_.compute_optimization_layer(cv_in, left_rgb, img_left_array.shape)
        cost_volumes_gt = optimization_.sgm_cpp(
            cv_in, invalid_value, confidence_is_int, p1_mat, p2_mat, optimization_layer, invalid_disp
        )
        # Invalid disparities of the cost volume as set as -9999
        cost_volumes_gt["cv"][invalid_disp] = -9999

        # Check if the calculated optimized cv is equal to the ground truth
        np.testing.assert_array_equal(cost_volumes_gt["cv"], out_cv["cost_volume"].data)


if __name__ == "__main__":
    unittest.main()
