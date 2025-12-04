# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
This module contains functions to test all plugin configurations.
"""

import json_checker
import pytest

from pandora import optimization

pytestmark = pytest.mark.usefixtures("import_plugin")


class TestCheckConf:
    """Test configuration"""

    @pytest.fixture()
    def optimization_cfg(self, use_confidence_value):
        """Test nominal configuration with use_confidence"""
        return {"optimization_method": "sgm", "use_confidence": use_confidence_value}

    @pytest.mark.parametrize(
        "use_confidence_value",
        [
            pytest.param(1, id="Integer"),
            pytest.param([1, 1], id="Integer list"),
            pytest.param((1, 1), id="Integer Tuple"),
            pytest.param(True, id="Boolean"),
            pytest.param({"test": 1}, id="Dict"),
        ],
    )
    def test_with_wrong_use_confidence_parameter(self, left_crafted, optimization_cfg):
        """Test with incorrect value for use_confidence parameter"""
        with pytest.raises(json_checker.core.exceptions.DictCheckerError):
            optimization.AbstractOptimization(left_crafted, **optimization_cfg)

    @pytest.mark.parametrize("use_confidence_value", ["cost_volume_confidence"])
    def test_with_nominal_use_confidence_parameter(self, left_crafted, optimization_cfg):
        """Test nominal value of use_confidence parameter"""
        optimization.AbstractOptimization(left_crafted, **optimization_cfg)

    @pytest.mark.parametrize("use_confidence_value", ["cost_volume_confidence"])
    def test_without_use_confidence_parameter(self, left_crafted, optimization_cfg):
        """Test without use_confidence parameter"""
        del optimization_cfg["use_confidence"]
        optimization.AbstractOptimization(left_crafted, **optimization_cfg)
