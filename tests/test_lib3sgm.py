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
from pandora import optimization, check_conf
from pandora.state_machine import PandoraMachine
import pandora.check_json as JSON_checker
from tests import common

# pylint: disable=too-many-lines, too-many-public-methods


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
        # Cones images with classification
        self.left_cones_classif = pandora.read_img(
            "tests/inputs/left.png", no_data=np.nan, mask=None, classif="tests/inputs/left_classif.tif"
        )
        self.right_cones_classif = pandora.read_img(
            "tests/inputs/right.png", no_data=np.nan, mask=None, classif="tests/inputs/right_classif.tif"
        )
        # Cones images with segmentation
        self.left_cones_segm = pandora.read_img(
            "tests/inputs/left.png", no_data=np.nan, mask=None, segm="tests/inputs/left_classif.tif"
        )
        self.right_cones_segm = pandora.read_img(
            "tests/inputs/right.png", no_data=np.nan, mask=None, segm="tests/inputs/right_classif.tif"
        )

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
                "indicator": ["confidence_from_ambiguity"],
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
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

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

        classif_arr = optimization_.compute_optimization_layer(
            self.cv, self.left_crafted, self.left_crafted["im"].data.shape
        )

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

        prior_array_out = optimization_.compute_optimization_layer(cv_in, left, left["im"].data.shape)

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
        left, right = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

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
            _, _ = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

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
            _, _ = pandora.run(pandora_machine, self.left_cones, self.right_cones, -60, 0, user_cfg)

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

        left, right = pandora.run(pandora_machine, masked_left, masked_right, -60, 0, user_cfg)

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

    def test_classif_on_right_and_left_with_one_class(self):
        """
        Optimization on one existing band for left and right classification with validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and one class
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["olive tree"]}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, self.left_cones_classif, self.right_cones_classif, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open("tests/outputs/right_disparity_3sgm.tif").read(1)

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

    def test_classif_on_right_and_left_with_two_classes(self):
        """
        Optimization on two existing bands for left and right classification with validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and two classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree", "cornfields"],
        }

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, self.left_cones_classif, self.right_cones_classif, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open("tests/outputs/right_disparity_3sgm.tif").read(1)

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

    def test_classif_on_right_and_left_with_wrong_class(self):
        """
        Optimization on wrong band for left and right classification with validation step.
        "peuplier" band doesn't exists.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and a wrong class
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["peuplier"]}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_right_and_left_with_no_class(self):
        """
        Optimization for left and right classification with wrong configuration with validation step.
        Classes are required for source as "classif" in geometric_prior.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_right_with_validation(self):
        """
        Optimization with only right classification present
        Validation step requires both left and right classifications
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["cornfields"]}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_validation(self):
        """
        Optimization for left classification with validation step.
        Validation step requires both left and right classifications
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["cornfields"]}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_right_and_left(self):
        """
        Optimization left and right segmentation with validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, self.left_cones_segm, self.right_cones_segm, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open("tests/outputs/right_disparity_3sgm.tif").read(1)

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

    def test_segm_with_classes(self):
        """
        Optimization left and right segmentation with validation step and classes instantiated.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm", "classes": ["cornfields"]}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_right(self):
        """
        Optimization right segmentation with validation step.
        Validation step requires both left and right segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left(self):
        """
        Optimization left segmentation with validation step.
        Validation step requires both left and right segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_correct_class(self):
        """
        Optimization on one existing band left classification without validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["olive tree"]}

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, self.left_cones_classif, self.right_cones, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

    def test_classif_on_left_and_right_with_correct_class(self):
        """
        Optimization on one existing band for left and right classification without validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["olive tree"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, self.left_cones_classif, self.right_cones_classif, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

    def test_classif_on_right_with_correct_class(self):
        """
        Optimization with right classification without validation step.
        Classification without validation step requires left classification.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["olive tree"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_without_validation(self):
        """
        Optimization on left image with segmentation without validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, self.left_cones_segm, self.right_cones, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

    def test_segm_on_left_and_right_without_validation(self):
        """
        Optimization on left and right image with segmentation without validation step.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, self.left_cones_segm, self.right_cones_segm, -60, 0, user_cfg)

        # Ground truth
        gt_left = rasterio.open("tests/outputs/left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if common.error(left["disparity_map"].data, gt_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if common.error(left["disparity_map"].data, gt_left, 2) > 0.15:
            raise AssertionError

    def test_segm_on_right_without_validation(self):
        """
        Optimization with right segmentation without validation step.
        Segmentation without validation step requires left segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/left_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_and_right_with_classes(self):
        """
        Optimization with right and left segmentation with classes without validation step.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm", "classes": ["olive tree"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_with_classes(self):
        """
        Optimization with left segmentation with classes without validation step.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm", "classes": ["olive tree"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_segm": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_segm": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_false_classes(self):
        """
        Optimization with left classification with false classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["pine"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_without_classes(self):
        """
        Optimization with left classification with no classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_and_right_with_wrong_classes(self):
        """
        Optimization with left and right classification with wrong classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif", "classes": ["pine"]}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_and_right_without_classes(self):
        """
        Optimization with left and right classification without classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"
        user_cfg["pipeline"]["right_disp_map"]["method"] = "none"

        # Add inputs
        user_cfg["input"] = {
            "img_left": "tests/inputs/left.png",
            "left_classif": "tests/inputs/left_classif.tif",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/right_classif.tif",
            "disp_min": -60,
            "disp_max": 0,
            "nodata_left": "NaN",
            "nodata_right": "NaN",
        }

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_user_initiate_3sgm_and_validation_with_one_geomprior_segmentation(self):
        """
        Test that user can't implement 3SGM and validation if only one segmentation is given
        """

        # Prepare the SGM configuration. It contains cross_checking validation and 3SGM optimization
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Create input cfg where only the left segmentation is present
        input_cfg = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "left_segm": "tests/inputs/white_band_mask.png",
            "disp_min": -60,
            "disp_max": 0,
        }

        # Add a segmentation geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}

        # Create full configuration
        cfg = {"input": input_cfg, "pipeline": user_cfg["pipeline"]}

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        self.assertRaises(SystemExit, JSON_checker.check_conf, cfg, pandora_machine)

    def test_user_initiate_3sgm_and_validation_with_one_geomprior_classification(self):
        """
        Test that user can't implement 3SGM and validation if only one classification is given
        """

        # Prepare the SGM configuration. It contains cross_checking validation
        user_cfg = pandora.read_config_file("tests/conf/3sgm.json")

        # Create input cfg where only the right classification is present
        input_cfg = {
            "img_left": "tests/inputs/left.png",
            "img_right": "tests/inputs/right.png",
            "right_classif": "tests/inputs/white_band_mask.png",
            "disp_min": -60,
            "disp_max": 0,
        }

        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}

        # Create full configuration
        cfg = {"input": input_cfg, "pipeline": user_cfg["pipeline"]}

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        self.assertRaises(SystemExit, JSON_checker.check_conf, cfg, pandora_machine)


if __name__ == "__main__":
    unittest.main()
