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
import pytest
import numpy as np
import pandora
import rasterio
import xarray as xr
from pandora import optimization
from pandora.state_machine import PandoraMachine

from tests import common


class TestPlugin3SGM(unittest.TestCase):
    """
    TestPlugin class allows to test pandora + plugin_lib3sgm
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
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )

        data = np.array(([1, 1, 1, 2, 2], [1, 1, 1, 4, 2], [1, 1, 1, 4, 4], [1, 1, 1, 1, 1]), dtype=np.float32)
        self.right_crafted = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
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

    def test_lib3sgm(self):
        """
        Test pandora + plugin_lib3sgm
        """
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg["pipeline"])

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

    def test_compute_optimization_layer_in_dataset(self):
        """
        Test plugin_lib3sgm compute_optimization_layer function,
        with user asking for piecewise optimization, present in image dataset
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # add internal geometric_prior source
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}

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

        classif_arr = optimization_.compute_optimization_layer(self.cv, img_left)

        gt_classif = np.array(([[2, 1], [1, 3], [2, 2.6]]), dtype=np.float32)
        np.testing.assert_array_equal(classif_arr, gt_classif)

    def test_compute_optimization_layer_none_layer_in_dataset(self):
        """
        Test plugin_libsgm compute_optimization_layer function,
        with user asking for piecewise optimization, without any in dataset
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # add internal geometric_prior source
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        classif_arr = optimization_.compute_optimization_layer(self.cv, self.left_crafted)

        gt_classif = np.ones((4, 5))
        np.testing.assert_array_equal(classif_arr, gt_classif)

    def test_optimization_layer_with_3sgm(self):
        """
        Test the optimization layer function with 3sgm default configuration
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Load plugins
        optimization_ = optimization.AbstractOptimization(**user_cfg["pipeline"]["optimization"])

        data = np.array(([1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 4, 1]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={"no_data_img": 0, "valid_pixels": 0, "no_data_mask": 1, "crs": None, "transform": None},
        )
        gt_default_prior_array = np.ones(left["im"].shape, dtype=np.float32)

        cv_in = copy.deepcopy(self.cv)

        prior_array_out = optimization_.compute_optimization_layer(cv_in, left)

        # check that added array in cv is correct
        np.testing.assert_array_equal(cv_in["internal"], gt_default_prior_array)

        # Check that prior array is the default one
        np.testing.assert_array_equal(gt_default_prior_array, prior_array_out)

    def test_user_initiate_3sgm_with_geomprior_internal(self):
        """
        Test that user can implement internal geometric_prior with a 3sgm configuration
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg["pipeline"])

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

    def test_user_initiate_3sgm_with_none_geomprior_classif(self):
        """
        Test that user can't implement classif geometric_prior with a 3sgm configuration
        and no classification in image dataset
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg["pipeline"])

    def test_user_initiate_3sgm_with_none_geomprior_segmentation(self):
        """
        Test that user can't implement segmentation geometric_prior with a 3sgm configuration
        and no segmentation in image dataset
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg["pipeline"])

    @staticmethod
    def test_user_initiate_3sgm_with_geomprior_segmentation():
        """
        Test that user can implement segmentation geometric_prior with a 3sgm configuration
        and segmentation in image dataset
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add mask to left data
        masked_left = pandora.read_img(
            "tests/inputs/left.png", no_data=np.nan, mask=None, segm="tests/inputs/white_band_mask.png"
        )
        masked_right = pandora.read_img(
            "tests/inputs/right.png", no_data=np.nan, mask=None, segm="tests/inputs/white_band_mask.png"
        )

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open("tests/outputs/right_disparity_3sgm.tif").read(1)

        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        left, right = pandora.run(pandora_machine, masked_left, masked_right, -60, 0, user_cfg["pipeline"])

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(right["disparity_map"].data, gt_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(right["disparity_map"].data, gt_right, 2) > 0.15:
            raise AssertionError

    @staticmethod
    def test_user_initiate_3sgm_with_geomprior_classif():
        """
        Test that user can implement classification geometric_prior with a 3sgm configuration
        and classification in image dataset
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add mask to left data
        masked_left = pandora.read_img(
            "tests/inputs/left.png", no_data=np.nan, mask=None, classif="tests/inputs/white_band_mask.png"
        )
        masked_right = pandora.read_img(
            "tests/inputs/right.png", no_data=np.nan, mask=None, classif="tests/inputs/white_band_mask.png"
        )

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open("tests/outputs/right_disparity_3sgm.tif").read(1)

        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        left, right = pandora.run(pandora_machine, masked_left, masked_right, -60, 0, user_cfg["pipeline"])

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(right["disparity_map"].data, gt_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(right["disparity_map"].data, gt_right, 2) > 0.15:
            raise AssertionError


if __name__ == "__main__":
    unittest.main()
