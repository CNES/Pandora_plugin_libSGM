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

import rasterio
import unittest
import numpy as np
import xarray as xr

import pandora
from pandora import stereo, optimization
from pandora.state_machine import PandoraMachine


class TestPlugin(unittest.TestCase):
    """
    TestPlugin class allows to test pandora + plugin_libsgm
    """
    def setUp(self):
        """
        Method called to prepare the test fixture

        """
        self.left = pandora.read_img('tests/left.png', no_data=np.nan, cfg={'nodata1': 'np.nan', 'nodata2': 'np.nan',
                                                                          'valid_pixels': 0, 'no_data': 1}, mask=None)
        self.right = pandora.read_img('tests/right.png', no_data=np.nan, cfg={'nodata1': 'np.nan', 'nodata2': 'np.nan',
                                                                          'valid_pixels': 0, 'no_data': 1}, mask=None)
        self.disp_left = rasterio.open('tests/disp_left.tif').read(1)
        self.disp_right = rasterio.open('tests/disp_right.tif').read(1)
        self.occlusion = rasterio.open('tests/occl.png').read(1)

        self.disp_left_zncc = rasterio.open('tests/disp_left_zncc.tif').read(1)
        self.disp_right_zncc = rasterio.open('tests/disp_right_zncc.tif').read(1)

    def error(self, data, gt, threshold, unknown_disparity=0):
        """
        Percentage of bad pixels whose error is > threshold

        """
        row, col = data.shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if gt[r, c] != unknown_disparity:
                    if abs((data[r, c] + gt[r, c])) > threshold:
                        nb_error += 1

        return nb_error / float(row * col)

    def error_mask(self, data, gt):
        """
        Percentage of bad pixels ( != ground truth ) in the validity mask

        """
        row, col = data.shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if data[r, c] != gt[r, c]:
                    nb_error += 1

        return nb_error / float(row * col)

    def strict_error(self, data, gt):
        """
        Average of bad pixels  ( != ground truth )

        """
        row, col = data.shape
        nb_error = 0
        for r in range(row):
            for c in range(col):
                if data[r, c] != gt[r, c]:
                    nb_error += 1

        return nb_error / float(row * col)

    def test_libsgm(self):
        """
        Test pandora + plugin_libsgm

        """
        user_cfg = pandora.read_config_file('conf/sgm.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left['validity_mask'].shape[0], left['validity_mask'].shape[1]))
        occlusion[left['validity_mask'].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if self.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_negative_disparities(self):
        """
        Test pandora + plugin_libsgm, with negative disparities

        """
        user_cfg = pandora.read_config_file('conf/sgm.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, -1, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Check the left validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((left['validity_mask'].shape[0], left['validity_mask'].shape[1]))
        occlusion[left['validity_mask'].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if self.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_positive_disparities(self):
        """
        Test pandora + plugin_libsgm, with positive disparities

        """
        user_cfg = pandora.read_config_file('conf/sgm.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        right, left = pandora.run(pandora_machine, self.right, self.left, 1, 60, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 1) > 0.20:
            raise AssertionError

        # Compares the calculated left disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(left['disparity_map'].data, self.disp_left, 2) > 0.15:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 1) > 0.20:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * right['disparity_map'].data, self.disp_right, 2) > 0.15:
            raise AssertionError

    def test_libsgm_zncc(self):
        """
        Test pandora + plugin_libsgm if ZNCC measure is used
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file('conf/sgm_zncc.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        left, right = pandora.run(pandora_machine, self.left, self.right, -60, 0, user_cfg)

        # Compares the calculated left disparity map with the ground truth
        # If the disparity maps are not equal, raise an error

        if self.strict_error(left['disparity_map'].data, self.disp_left_zncc) > 0:
            raise AssertionError

        # Compares the calculated right disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        if self.strict_error(right['disparity_map'].data, self.disp_right_zncc) > 0:
            raise AssertionError

    def test_number_of_disp(self):
        """
        Test plugin_libsgm number_of_disp function if min_cost_paths is activated
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file('conf/sgm.json')
        user_cfg['pipeline']['stereo']['window_size'] = 3
        user_cfg['pipeline']['optimization']['min_cost_paths'] = True

        # Load plugins
        stereo_ = stereo.AbstractStereo(**user_cfg['pipeline']['stereo'])
        optimization_ = optimization.AbstractOptimization(**user_cfg['pipeline']['optimization'])

        # Import pandora plugins
        pandora.import_plugin()

        data = np.array(([1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 2],
                         [1, 1, 1, 4, 3],
                         [1, 1, 1, 1, 1]), dtype=np.float32)
        left = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])},
                         attrs={'no_data_img': 0, 'valid_pixels': 0, 'no_data_mask': 1})

        data = np.array(([1, 1, 1, 2, 2],
                         [1, 1, 1, 4, 2],
                         [1, 1, 1, 4, 4],
                         [1, 1, 1, 1, 1]), dtype=np.float32)
        right = xr.Dataset({'im': (['row', 'col'], data)},
                         coords={'row': np.arange(data.shape[0]), 'col': np.arange(data.shape[1])},
                         attrs={'no_data_img': 0, 'valid_pixels': 0, 'no_data_mask': 1})

        # Computes the cost volume dataset
        cv = stereo_.compute_cost_volume(img_left=left, img_right=right, disp_min=-2, disp_max=2)

        # Disparities which give a minimum local cost, in indices
        disp_path = np.array([[[1, 3, 2, 3, 1, 3, 3, 2],
                               [0, 1, 1, 4, 2, 2, 3, 1],
                               [2, 4, 2, 4, 3, 0, 3, 3]],
                              [[ 3, 3, 2, 3, 1, 0, 1, 3],
                               [ 2, 1, 1, 3, 1, 3, 1, 2],
                               [0, 0, 0, 0, 0, 0, 0, 0]]], dtype=np.float32)

        invalid_disp = np.isnan(cv['cost_volume'].data)
        cv_updated = optimization_.number_of_disp(cv, disp_path, invalid_disp)

        # Ground truth calculated with disp_path
        gt_disp = np.array([[2, 3, 0],
                            [1, 4, 8]], dtype=np.float32)

        # Check if the calculated confidence_measure is equal to the ground truth (same shape and all elements equals)
        np.testing.assert_array_equal(cv_updated['confidence_measure'].data[:, :, -1], gt_disp)


if __name__ == '__main__':
    unittest.main()
