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
This module provides functions to test Pandora + plugin_LibSGM
"""

import copy
import pytest
import numpy as np
import pandora
import rasterio
import xarray as xr
from transitions import MachineError
from pandora import optimization
from pandora.check_configuration import check_conf
from pandora.margins import Margins
from pandora.state_machine import PandoraMachine
from tests import common


# pylint: disable=too-many-lines, too-many-public-methods, redefined-outer-name

pytestmark = pytest.mark.usefixtures("import_plugin")


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


@pytest.fixture()
def user_cfg(configurations_path):
    return pandora.read_config_file(str(configurations_path / "3sgm.json"))


@pytest.fixture()
def inputs_with_classif(inputs_path):
    return {
        "left": {
            "img": str(inputs_path / "left.png"),
            "classif": str(inputs_path / "left_classif.tif"),
            "disp": [-60, 0],
            "nodata": "NaN",
        },
        "right": {
            "img": str(inputs_path / "right.png"),
            "classif": str(inputs_path / "right_classif.tif"),
            "nodata": "NaN",
        },
    }


@pytest.fixture()
def inputs_with_segment(inputs_path):
    return {
        "left": {
            "img": str(inputs_path / "left.png"),
            "segm": str(inputs_path / "left_classif.tif"),
            "disp": [-60, 0],
            "nodata": "NaN",
        },
        "right": {
            "img": str(inputs_path / "right.png"),
            "segm": str(inputs_path / "right_classif.tif"),
            "nodata": "NaN",
        },
    }


@pytest.fixture()
def monoband_image():
    """Return a 2D dataset builder of a given shape."""

    def inner(shape):
        return xr.Dataset(
            {},
            coords={"band_im": [None], "row": np.arange(shape[0]), "col": np.arange(shape[1])},
            attrs={"disparity_source": None},
        )

    return inner


@pytest.fixture()
def pandora_machine_builder():
    """Return a pandora machine builder which expects an image shape."""

    def builder(image_shape, image_builder):
        machine = PandoraMachine()
        machine.left_img = image_builder(image_shape)
        machine.right_img = image_builder(image_shape)
        return machine

    return builder


class TestPlugin3SGM:
    """
    TestPlugin class allows to test pandora + plugin_lib3sgm
    """

    @pytest.mark.parametrize(
        ["image_shape", "configuration", "expected"],
        [
            pytest.param(
                (10, 10),
                {
                    "pipeline": {
                        "matching_cost": {
                            "matching_cost_method": "zncc",
                            "window_size": 11,
                            "subpix": 2,
                            "band": None,
                            "step": 1,
                        },  # Margins(5, 5, 5, 5)
                        "optimization": {"optimization_method": "sgm"},  # Margins(40, 40, 40, 40)
                    },
                },
                Margins(45, 45, 45, 45),
                id="Only matching_cost margins with optimization",
            )
        ],
    )
    def test_margins(self, pandora_machine_builder, image_shape, monoband_image, configuration, expected):
        """
        Given a pipeline with steps, each step with margins should contribute to global margins.
        """
        # NOTE: actual code is in Pandora, not Pandora2D
        pandora_machine = pandora_machine_builder(image_shape, monoband_image)
        pandora_machine.check_conf(configuration, pandora_machine.left_img, pandora_machine.right_img)

        assert pandora_machine.margins.global_margins == expected

    def test_lib3sgm(self, left_cones, right_cones, disp_left, disp_right, user_cfg):
        """
        Test pandora + plugin_lib3sgm
        """
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

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

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 2) <= 0.15

    def test_compute_optimization_layer_none_layer_in_dataset(self, left_crafted, cost_volume, user_cfg):
        """
        Test plugin_libsgm compute_optimization_layer function,
        with user asking for piecewise optimization, without any in dataset
        """

        # Prepare the configuration
        # add internal geometric_prior source
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left_crafted, **user_cfg["pipeline"]["optimization"])

        classif_arr = optimization_.compute_optimization_layer(cost_volume, left_crafted, left_crafted["im"].data.shape)

        gt_classif = np.ones((4, 5))
        np.testing.assert_array_equal(classif_arr, gt_classif)

    def test_optimization_layer_with_3sgm(self, cost_volume, user_cfg):
        """
        Test the optimization layer function with 3sgm default configuration
        """

        data = np.array(([1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 4, 1]), dtype=np.float32)
        left = xr.Dataset(
            {"im": (["row", "col"], data)},
            coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
            attrs={
                "no_data_img": 0,
                "valid_pixels": 0,
                "no_data_mask": 1,
                "crs": None,
                "transform": None,
            },
        )
        gt_default_prior_array = np.ones(left["im"].shape, dtype=np.float32)

        # Load plugins
        optimization_ = optimization.AbstractOptimization(left, **user_cfg["pipeline"]["optimization"])

        cv_in = copy.deepcopy(cost_volume)

        prior_array_out = optimization_.compute_optimization_layer(cv_in, left, left["im"].data.shape)

        # check that added array in cv is correct
        np.testing.assert_array_equal(cv_in["internal"], gt_default_prior_array)

        # Check that prior array is the default one
        np.testing.assert_array_equal(gt_default_prior_array, prior_array_out)

    def test_user_initiate_3sgm_with_geomprior_internal(self, left_cones, right_cones, disp_left, disp_right, user_cfg):
        """
        Test that user can implement internal geometric_prior with a 3sgm configuration
        """

        # Prepare the SGM configuration
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "internal"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

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

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, disp_right, 2) <= 0.15

    def test_user_initiate_3sgm_with_none_geomprior_classif(self, left_cones, right_cones, user_cfg):
        """
        Test that user can't implement classif geometric_prior with a 3sgm configuration
        and no classification in image dataset
        """

        # Prepare the SGM configuration
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

    def test_user_initiate_3sgm_with_none_geomprior_segmentation(self, left_cones, right_cones, user_cfg):
        """
        Test that user can't implement segmentation geometric_prior with a 3sgm configuration
        and no segmentation in image dataset
        """

        # Prepare the SGM configuration
        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Pandora pipeline should fail
        with pytest.raises(SystemExit):
            _, _ = pandora.run(pandora_machine, left_cones, right_cones, user_cfg)

    @staticmethod
    def test_user_initiate_3sgm_with_geomprior_segmentation(user_cfg, inputs_path, outputs_path):
        """
        Test that user can implement segmentation geometric_prior with a 3sgm configuration
        and segmentation in image dataset
        """

        # Add mask to left data
        masked_left = pandora.create_dataset_from_inputs(
            {
                "img": str(inputs_path / "left.png"),
                "nodata": np.nan,
                "mask": None,
                "segm": str(inputs_path / "white_band_mask.png"),
                "disp": [-60, 0],
            }
        )
        masked_right = pandora.create_dataset_from_inputs(
            {
                "img": str(inputs_path / "right.png"),
                "nodata": np.nan,
                "mask": None,
                "segm": str(inputs_path / "white_band_mask.png"),
                "disp": [0, 60],
            }
        )

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open(outputs_path / "right_disparity_3sgm.tif").read(1)

        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Instantiate machine
        pandora_machine = PandoraMachine()

        left, right = pandora.run(pandora_machine, masked_left, masked_right, user_cfg)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 2) <= 0.15

    def test_classif_on_right_and_left_with_one_class(
        self, left_cones_classif, right_cones_classif, user_cfg, inputs_with_classif, outputs_path
    ):
        """
        Optimization on one existing band for left and right classification with validation step.
        """

        # Prepare the SGM configuration
        # Add a classification and one class
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree"],
        }

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, left_cones_classif, right_cones_classif, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open(outputs_path / "right_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 2) <= 0.15

    def test_classif_on_right_and_left_with_two_classes(
        self, left_cones_classif, right_cones_classif, user_cfg, inputs_with_classif, outputs_path
    ):
        """
        Optimization on two existing bands for left and right classification with validation step.
        """

        # Prepare the SGM configuration
        # Add a classification and two classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree", "cornfields"],
        }

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, left_cones_classif, right_cones_classif, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open(outputs_path / "right_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 2) <= 0.15

    def test_classif_on_right_and_left_with_wrong_class(self, user_cfg, inputs_with_classif):
        """
        Optimization on wrong band for left and right classification with validation step.
        "peuplier" band doesn't exists.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and a wrong class
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["peuplier"],
        }

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(MachineError, match="A problem occurs during Pandora checking. Be sure of your sequencing"):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_right_and_left_with_no_class(self, user_cfg, inputs_with_classif):
        """
        Optimization for left and right classification with wrong configuration with validation step.
        Classes are required for source as "classif" in geometric_prior.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_right_with_validation(self, user_cfg, inputs_with_classif):
        """
        Optimization with only right classification present
        Validation step requires both left and right classifications
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["cornfields"],
        }
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["left"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_validation(self, user_cfg, inputs_with_classif):
        """
        Optimization for left classification with validation step.
        Validation step requires both left and right classifications
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["cornfields"],
        }
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["right"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_right_and_left(
        self, left_cones_segm, right_cones_segm, user_cfg, inputs_with_segment, outputs_path
    ):
        """
        Optimization left and right segmentation with validation step.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}

        # Add inputs
        user_cfg["input"] = inputs_with_segment

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, right = pandora.run(pandora_machine, left_cones_segm, right_cones_segm, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)
        gt_right = rasterio.open(outputs_path / "right_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(right["disparity_map"].data, gt_right, 2) <= 0.15

    def test_segm_with_classes(self, user_cfg, inputs_with_classif):
        """
        Optimization left and right segmentation with validation step and classes instantiated.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "segm",
            "classes": ["cornfields"],
        }

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_right(self, user_cfg, inputs_with_segment):
        """
        Optimization right segmentation with validation step.
        Validation step requires both left and right segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_segment
        del user_cfg["input"]["left"]["segm"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left(self, user_cfg, inputs_with_segment):
        """
        Optimization left segmentation with validation step.
        Validation step requires both left and right segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_segment
        del user_cfg["input"]["right"]["segm"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_correct_class(
        self, left_cones_classif, right_cones, user_cfg, inputs_with_classif, outputs_path
    ):
        """
        Optimization on one existing band left classification without validation step.
        """

        # Prepare the SGM configuration
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree"],
        }

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["right"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, left_cones_classif, right_cones, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

    def test_classif_on_left_and_right_with_correct_class(
        self, left_cones_classif, right_cones_classif, user_cfg, inputs_with_classif, outputs_path
    ):
        """
        Optimization on one existing band for left and right classification without validation step.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, left_cones_classif, right_cones_classif, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

    def test_classif_on_right_with_correct_class(self, user_cfg, inputs_with_classif):
        """
        Optimization with right classification without validation step.
        Classification without validation step requires left classification.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["olive tree"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["left"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_without_validation(
        self, left_cones_segm, right_cones, user_cfg, inputs_with_segment, outputs_path
    ):
        """
        Optimization on left image with segmentation without validation step.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_segment
        del user_cfg["input"]["right"]["segm"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, left_cones_segm, right_cones, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

    def test_segm_on_left_and_right_without_validation(
        self, left_cones_segm, right_cones_segm, user_cfg, inputs_with_segment, outputs_path
    ):
        """
        Optimization on left and right image with segmentation without validation step.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_segment

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Check configuration
        user_cfg = check_conf(user_cfg, pandora_machine)

        left, _ = pandora.run(pandora_machine, left_cones_segm, right_cones_segm, user_cfg)

        # Ground truth
        gt_left = rasterio.open(outputs_path / "left_disparity_3sgm.tif").read(1)

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 1) <= 0.20

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        assert common.error(left["disparity_map"].data, gt_left, 2) <= 0.15

    def test_segm_on_right_without_validation(self, user_cfg, inputs_with_segment):
        """
        Optimization with right segmentation without validation step.
        Segmentation without validation step requires left segmentation.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]
        # Because the pipeline isn't correctly checked in this test
        user_cfg["pipeline"]["disparity"]["invalid_disparity"] = np.nan

        # Add inputs
        user_cfg["input"] = inputs_with_segment
        del user_cfg["input"]["left"]["segm"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_and_right_with_classes(self, user_cfg, inputs_with_segment):
        """
        Optimization with right and left segmentation with classes without validation step.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "segm",
            "classes": ["olive tree"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_segment

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_segm_on_left_with_classes(self, user_cfg, inputs_with_segment):
        """
        Optimization with left segmentation with classes without validation step.
        Classes are not available for segmentation step
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a segmentation and classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "segm",
            "classes": ["olive tree"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_segment

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_with_false_classes(self, user_cfg, inputs_with_classif):
        """
        Optimization with left classification with false classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["pine"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["right"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(MachineError, match="A problem occurs during Pandora checking. Be sure of your sequencing"):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_without_classes(self, user_cfg, inputs_with_classif):
        """
        Optimization with left classification with no classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_classif
        del user_cfg["input"]["right"]["classif"]

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_and_right_with_wrong_classes(self, user_cfg, inputs_with_classif):
        """
        Optimization with left and right classification with wrong classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {
            "source": "classif",
            "classes": ["pine"],
        }
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(MachineError, match="A problem occurs during Pandora checking. Be sure of your sequencing"):
            _ = check_conf(user_cfg, pandora_machine)

    def test_classif_on_left_and_right_without_classes(self, user_cfg, inputs_with_classif):
        """
        Optimization with left and right classification without classes without validation step.
        Check that the check_conf function raises an error.
        """

        # Prepare the SGM configuration
        # Add a classification and false classes
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}
        # Remove validation step
        del user_cfg["pipeline"]["validation"]
        del user_cfg["pipeline"]["filter.after.validation"]

        # Add inputs
        user_cfg["input"] = inputs_with_classif

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(user_cfg, pandora_machine)

    def test_user_initiate_3sgm_and_validation_with_one_geomprior_segmentation(self, user_cfg, inputs_path):
        """
        Test that user can't implement 3SGM and validation if only one segmentation is given
        """

        # Prepare the SGM configuration. It contains cross_checking validation and 3SGM optimization
        # Create input cfg where only the left segmentation is present
        input_cfg = {
            "left": {
                "img": str(inputs_path / "left.png"),
                "segm": str(inputs_path / "white_band_mask.png"),
                "disp": [-60, 0],
            },
            "right": {
                "img": str(inputs_path / "right.png"),
            },
        }

        # Add a segmentation geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "segm"}

        # Create full configuration
        cfg = {"input": input_cfg, "pipeline": user_cfg["pipeline"]}

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(cfg, pandora_machine)

    def test_user_initiate_3sgm_and_validation_with_one_geomprior_classification(self, user_cfg, inputs_path):
        """
        Test that user can't implement 3SGM and validation if only one classification is given
        """

        # Prepare the SGM configuration. It contains cross_checking validation
        # Create input cfg where only the right classification is present
        input_cfg = {
            "left": {
                "img": str(inputs_path / "left.png"),
                "disp": [-60, 0],
            },
            "right": {
                "img": str(inputs_path / "right.png"),
                "classif": str(inputs_path / "white_band_mask.png"),
            },
        }

        # Add a geometric_prior
        user_cfg["pipeline"]["optimization"]["geometric_prior"] = {"source": "classif"}

        # Create full configuration
        cfg = {"input": input_cfg, "pipeline": user_cfg["pipeline"]}

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # check the configuration
        with pytest.raises(SystemExit):
            _ = check_conf(cfg, pandora_machine)
