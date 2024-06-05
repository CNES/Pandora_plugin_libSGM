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
import pytest
import numpy as np

import pandora
from pandora.state_machine import PandoraMachine
from tests import common

pytestmark = pytest.mark.usefixtures("import_plugin")


class TestPluginPythonParall:
    """
    TestPlugin class allows to test pandora + plugin_libsgm
    """

    def test_libsgm(self, left_cones, right_cones, disp_left, disp_right):
        """
        Test pandora + plugin_libsgm

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, -60, 0, user_cfg)

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

    def test_libsgm_negative_disparities(self, left_cones, right_cones, disp_left, disp_right):
        """
        Test pandora + plugin_libsgm, with negative disparities

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, -60, -1, user_cfg)

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

    def test_libsgm_positive_disparities(self, left_cones, right_cones, disp_left, disp_right):
        """
        Test pandora + plugin_libsgm, with positive disparities

        """
        user_cfg = pandora.read_config_file("tests/conf/sgm_python_parall.json")

        # Instantiate machine
        pandora_machine = PandoraMachine()

        right, left = pandora.run(pandora_machine, right_cones, left_cones, 1, 60, user_cfg)

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

    def test_libsgm_zncc(self, left_cones, right_cones, disp_left_zncc, disp_right_zncc):
        """
        Test pandora + plugin_libsgm if ZNCC measure is used
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file("tests/conf/sgm_zncc_python_parall.json")

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, left_cones, right_cones, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        np.testing.assert_allclose(left["disparity_map"].data, disp_left_zncc, rtol=1e-04)

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        np.testing.assert_allclose(right["disparity_map"].data, disp_right_zncc, rtol=1e-04)
