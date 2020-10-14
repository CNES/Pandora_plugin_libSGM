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
        self.ref = pandora.read_img('tests/ref.png', no_data=np.nan, cfg={'nodata1': 'np.nan', 'nodata2': 'np.nan',
                                                                          'valid_pixels': 0, 'no_data': 1}, mask=None)
        self.sec = pandora.read_img('tests/sec.png', no_data=np.nan, cfg={'nodata1': 'np.nan', 'nodata2': 'np.nan',
                                                                          'valid_pixels': 0, 'no_data': 1}, mask=None)
        self.disp_ref = rasterio.open('tests/disp_ref.tif').read(1)
        self.disp_sec = rasterio.open('tests/disp_sec.tif').read(1)
        self.occlusion = rasterio.open('tests/occl.png').read(1)

        self.disp_ref_zncc = rasterio.open('tests/disp_ref_zncc.tif').read(1)
        self.disp_sec_zncc = rasterio.open('tests/disp_sec_zncc.tif').read(1)

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
        user_cfg = pandora.read_config_file('conf/sgm_python.json')

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Import pandora plugins
        pandora.import_plugin()

        # Run the pandora pipeline
        ref, sec = pandora.run(pandora_machine, self.ref, self.sec, -60, 0, user_cfg)

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 1) > 0.20:
            raise AssertionError

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 2) > 0.15:
            raise AssertionError

        # Check the reference validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((ref['validity_mask'].shape[0], ref['validity_mask'].shape[1]))
        occlusion[ref['validity_mask'].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if self.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 1) > 0.20:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 2) > 0.15:
            raise AssertionError

    def test_libsgm_negative_disparities(self):
        """
        Test pandora + plugin_libsgm, with negative disparities

        """
        user_cfg = pandora.read_config_file('conf/sgm_python.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        ref, sec = pandora.run(pandora_machine, self.ref, self.sec, -60, -1, user_cfg)

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 1) > 0.20:
            raise AssertionError

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 2) > 0.15:
            raise AssertionError

        # Check the reference validity mask cross checking ( bit 8 and 9 )
        # Compares the calculated validity mask with the ground truth ( occlusion mask )
        occlusion = np.ones((ref['validity_mask'].shape[0], ref['validity_mask'].shape[1]))
        occlusion[ref['validity_mask'].data >= 512] = 0

        # If the percentage of errors is > 0.15, raise an error
        if self.error_mask(occlusion, self.occlusion) > 0.15:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 1) > 0.20:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 2) > 0.15:
            raise AssertionError

    def test_libsgm_positive_disparities(self):
        """
        Test pandora + plugin_libsgm, with positive disparities

        """
        user_cfg = pandora.read_config_file('conf/sgm_python.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        sec, ref = pandora.run(pandora_machine, self.sec, self.ref, 1, 60, user_cfg)

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 1) > 0.20:
            raise AssertionError

        # Compares the calculated reference disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(ref['disparity_map'].data, self.disp_ref, 2) > 0.15:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors is > 0.20, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 1) > 0.20:
            raise AssertionError

        # Compares the calculated secondary disparity map with the ground truth
        # If the percentage of pixel errors ( error if ground truth - calculate > 2) is > 0.15, raise an error
        if self.error(-1 * sec['disparity_map'].data, self.disp_sec, 2) > 0.15:
            raise AssertionError

    def test_libsgm_zncc(self):
        """
        Test pandora + plugin_libsgm if ZNCC measure is used
        """

        # Prepare the configuration
        user_cfg = pandora.read_config_file('conf/sgm_zncc_python.json')

        # Import pandora plugins
        pandora.import_plugin()

        # Instantiate machine
        pandora_machine = PandoraMachine()

        # Run the pandora pipeline
        ref, sec = pandora.run(pandora_machine, self.ref, self.sec, -60, 0, user_cfg)

        # Compares the calculated reference disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        np.testing.assert_allclose(ref['disparity_map'].data, self.disp_ref_zncc, rtol=1e-04)

        # Compares the calculated secondary disparity map with the ground truth
        # If the disparity maps are not equal, raise an error
        np.testing.assert_allclose(sec['disparity_map'].data, self.disp_sec_zncc, rtol=1e-04)


if __name__ == '__main__':
    unittest.main()
