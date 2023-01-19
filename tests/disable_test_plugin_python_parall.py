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
import rasterio
import numpy as np

import pandora
from pandora.state_machine import PandoraMachine
from tests import common


class TestPluginPythonParall(unittest.TestCase):
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
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Import pandora plugins
        pandora.import_plugin()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg)

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
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, -1, user_cfg)

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
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        right, left = pandora.run(pandora_machine, self.right, self.left, 1, 60, user_cfg)

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
        user_cfg = pandora.read_config_file("tests/conf/sgm_zncc_python_parall.json")

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        np.testing.assert_allclose(left["disparity_map"].data, self.disp_left_zncc, rtol=1e-04)

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        np.testing.assert_allclose(right["disparity_map"].data, self.disp_right_zncc, rtol=1e-04)


if __name__ == "__main__":
    unittest.main()
