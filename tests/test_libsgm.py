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

import numpy as np
import pandora
import pytest
import xarray as xr
from pandora import matching_cost, optimization, cost_volume_confidence
from pandora.state_machine import PandoraMachine
from pandora_plugin_libsgm.abstract_sgm import get_band_values

from tests import common

# pylint: disable=redefined-outer-name, duplicate-code, fixme
# TODO: remove duplicated test with test_libsgm


@pytest.fixture()
def user_cfg(configurations_path):
    """Configuration fixture."""
    return pandora.read_config_file(str(configurations_path / "sgm.json"))


@pytest.fixture()
def user_zncc_cfg(configurations_path):
    """Configuration fixture."""
    return pandora.read_config_file(str(configurations_path / "sgm_zncc.json"))


@pytest.fixture()
def cost_volume():
    """Create cost volume."""
    data_cv = np.array(
        [
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]],
            [[1, 1, 2, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 7]],
            [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
        ],
        dtype=np.float32,
    )
    result = xr.Dataset(
        {"cost_volume": (["row", "col", "disp"], data_cv)},
        coords={
            "row": np.arange(data_cv.shape[0]),
            "col": np.arange(data_cv.shape[1]),
            "disp": np.arange(data_cv.shape[2]),
            "indicator": ["confidence_from_ambiguity"],
        },
        attrs={
            "no_data_img": 0,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": None,
        },
    )
    return result


class TestPluginSGM:
    """
    TestPlugin class allows to test pandora + plugin_libsgm
    """

    def test_libsgm(self, left_cones, right_cones, disp_left, disp_right, user_cfg):
        """
        Test pandora + plugin_libsgm

        """

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 1, flag_inverse_value=False) <= 0.20

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 2, flag_inverse_value=False) <= 0.15

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        assert common.error_mask(occlusion, occlusion) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 2) <= 0.15

    def test_libsgm_negative_disparities(
        self, left_cones, right_cones, disp_left, disp_right, user_cfg
    ):
        """
        Test pandora + plugin_libsgm, with negative disparities

        """

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 1, flag_inverse_value=False) <= 0.20

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 2, flag_inverse_value=False) <= 0.15

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left["validity_mask"].shape[0], left["validity_mask"].shape[1]))
        occlusion[left["validity_mask"].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        assert common.error_mask(occlusion, occlusion) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 2) <= 0.15

    def test_libsgm_positive_disparities(self, left_cones, right_cones, disp_left, disp_right, user_cfg):
        """
        Test pandora + plugin_libsgm, with positive disparities

        """

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        right, left = pandora.run(pandora_machine, right_cones, left_cones, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 1, flag_inverse_value=False) <= 0.20

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, disp_left, 2, flag_inverse_value=False) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 2) <= 0.15

    def test_libsgm_zncc(self, left_cones, right_cones, disp_left_zncc, disp_right_zncc, user_zncc_cfg):
        """
        Test pandora + plugin_libsgm if ZNCC measure is used
        """

        # Prepare the configuration
        user_cfg = user_zncc_cfg

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        assert (
            common.strict_error(
                left["disparity_map"].data[61:-61, 61:-61],
                disp_left_zncc[61:-61, 61:-61],
            )
            <= 0
        )

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        assert common.strict_error(right["disparity_map"].data, disp_right_zncc) <= 0

    def test_number_of_disp(self, left_crafted, right_crafted, user_cfg):
        """
        Test plugin_libsgm number_of_disp function if min_cost_paths is activated
        """

        # Prepare the configuration
        user_cfg["pipeline"]["matching_cost"]["window_size"] = 3
        user_cfg["pipeline"]["optimization"]["min_cost_paths"] = True

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        # Import pandora plugins
        pandora.import_plugin()

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(
            img_left=left_crafted,
            img_right=right_crafted,
            grid_disp_min=np.full((left_crafted.dims["row"], left_crafted.dims["col"]), -2),
            grid_disp_max=np.full((left_crafted.dims["row"], left_crafted.dims["col"]), 2),
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

    def test_number_of_disp_with_previous_confidence(self, left_crafted, right_crafted, user_cfg):
        """
        Test plugin_libsgm number_of_disp function if min_cost_paths is activated and the confidence measure was present
        """

        # Prepare the configuration
        user_cfg["pipeline"]["matching_cost"]["window_size"] = 3
        user_cfg["pipeline"]["optimization"]["min_cost_paths"] = True

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])
        confidence_ = cost_volume_confidence.AbstractCostVolumeConfidence(
            **user_cfg["pipeline"]["cost_volume_confidence"]
        )

        # Import pandora plugins
        pandora.import_plugin()

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(
            img_left=left_crafted,
            img_right=right_crafted,
            grid_disp_min=np.full((left_crafted.dims["row"], left_crafted.dims["col"]), -2),
            grid_disp_max=np.full((left_crafted.dims["row"], left_crafted.dims["col"]), 2),
        )
        _, cv = confidence_.confidence_prediction(None, left_crafted, right_crafted, cv)
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

    def test_apply_confidence_no_confidence(self, left_crafted, cost_volume, user_cfg):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        # Test
        use_confidence = False

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cost_volume, use_confidence)

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
        assert confidence_is_int

    def test_apply_confidence_no_confidence_dataarray(self, left_crafted, cost_volume, user_cfg):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        # Test
        use_confidence = True

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cost_volume, use_confidence)

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
        assert confidence_is_int

    def test_apply_confidence_with_confidence_dataarray(self, left_crafted, cost_volume, user_cfg):
        """
        Test plugin_libsgm apply_confidence function, with user asking for confidence usage, without any in dataser
        """

        # Prepare the configuration

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        # Test
        use_confidence = True

        data_confidence = np.expand_dims(
            np.array([[1, 1, 1, 0.5], [1, 1, 0.5, 1], [1, 1, 1, 1]], dtype=np.float32),
            axis=2,
        )

        cv_in = copy.deepcopy(cost_volume)
        cv_in["confidence_measure"] = xr.DataArray(data_confidence, dims=["row", "col", "indicator"])

        # apply confidence
        cv_updated, confidence_is_int = optimization_.apply_confidence(cv_in, use_confidence)

        # Ground Truth
        optim_cv_gt = np.array(
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 2],
                    [1, 1, 1, 4, 3],
                    [0.5, 0.5, 0.5, 0.5, 0.5],
                ],
                [
                    [1, 1, 2, 1, 1],
                    [1, 1, 1, 1, 2],
                    [0.5, 0.5, 0.5, 2, 1.5],
                    [1, 1, 1, 1, 7],
                ],
                [[1, 4, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 12, 1, 1, 1]],
            ],
            dtype=np.float32,
        )

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated["cost_volume"].data[:, :, :], optim_cv_gt)

        # Check if confidence_is_int is right
        assert confidence_is_int is False

    def test_optimization_layer_with_sgm(self, left_crafted, cost_volume, user_cfg):
        """
        Test the optimization layer function with sgm default configuration
        """

        # Prepare the configuration

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        cv_in = copy.deepcopy(cost_volume)

        prior_array_out = optimization_.compute_optimization_layer(cv_in, left_crafted, left_crafted["im"].shape)

        # Check that cost volume isn't changed
        with pytest.raises(KeyError):
            _ = cv_in["internal"]

        # Check that prior array is the default one
        default_prior_array = np.ones(left_crafted["im"].data.shape, dtype=np.float32)
        np.testing.assert_array_equal(default_prior_array, prior_array_out)

    def test_user_initiate_sgm_with_geomprior(self, left_cones, right_cones, user_cfg):
        """
        Test that user can't implement geometric_prior with a sgm configuration
        """

        # Prepare the SGM configuration
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

    def test_optimization_layer_with_multiband(self, user_cfg, left_rgb, right_rgb):
        """
        Test the optimization layer function with multiband input images
        """
        # Prepare the configuration for multiband
        user_cfg["pipeline"]["matching_cost"]["band"] = "g"
        # We choose the mccnn penalty to verify that the correct band is being used
        user_cfg["pipeline"]["optimization"]["penalty"]["penalty_method"] = "mc_cnn_fast_penalty"
        user_cfg["pipeline"]["optimization"]["penalty"].pop("p2_method")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        matching_cost_ = matching_cost.AbstractMatchingCost(**user_cfg["pipeline"]["matching_cost"])
        optimization_ = optimization.AbstractOptimization(left_rgb, **user_cfg["pipeline"]["optimization"])

        # Import pandora plugins
        pandora.import_plugin()

        # Computes the cost volume dataset
        cv = matching_cost_.compute_cost_volume(
            img_left=left_rgb,
            img_right=right_rgb,
            grid_disp_min=np.full((left_rgb.dims["row"], left_rgb.dims["col"]), -60),
            grid_disp_max=np.full((left_rgb.dims["row"], left_rgb.dims["col"]), 0),
        )
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
        (
            invalid_value,
            p1_mat,
            p2_mat,
        ) = optimization_._penalty.compute_penalty(  # pylint:disable=protected-access
            cv_in, img_left_array, img_right_array
        )
        cv_in, confidence_is_int = optimization_.apply_confidence(
            cv_in, optimization_._use_confidence  # pylint:disable=protected-access
        )
        optimization_layer = optimization_.compute_optimization_layer(cv_in, left_rgb, img_left_array.shape)
        cost_volumes_gt = optimization_.sgm_cpp(
            cv_in,
            invalid_value,
            confidence_is_int,
            p1_mat,
            p2_mat,
            optimization_layer,
            invalid_disp,
        )
        # Invalid disparities of the cost volume as set as -9999
        cost_volumes_gt["cv"][invalid_disp] = -9999

        # Check if the calculated optimized cv is equal to the ground truth
        np.testing.assert_array_equal(cost_volumes_gt["cv"], out_cv["cost_volume"].data)


@pytest.mark.parametrize(
    ["band_name", "expected"],
    [
        (None, np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]], dtype=np.float32)),
        ("r", np.array([[1, 1], [1, 1]], dtype=np.float32)),
        ("g", np.array([[2, 2], [2, 2]], dtype=np.float32)),
        ("b", np.array([[3, 3], [3, 3]], dtype=np.float32)),
    ],
)
def test_get_band_values(band_name, expected):
    """Given a band_name, test we get expected band values."""
    data = np.array(
        [
            [
                [1, 1],
                [1, 1],
            ],
            [
                [2, 2],
                [2, 2],
            ],
            [
                [3, 3],
                [3, 3],
            ],
        ],
        dtype=np.float32,
    )
    input_dataset = xr.Dataset(
        {"im": (["band_im", "row", "col"], data)},
        coords={
            "band_im": ["r", "g", "b"],
            "row": np.arange(2),
            "col": np.arange(2),
        },
    )

    result = get_band_values(input_dataset, band_name)

    np.testing.assert_array_equal(result, expected)
