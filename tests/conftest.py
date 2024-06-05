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
"""Fixtures."""
# pylint: disable=redefined-outer-name

import pandora
import pytest

import numpy as np
import rasterio
import xarray as xr


@pytest.fixture()
def inputs_path(resource_path_root):
    return resource_path_root / "inputs"


@pytest.fixture()
def outputs_path(resource_path_root):
    return resource_path_root / "outputs"


@pytest.fixture()
def configurations_path(resource_path_root):
    return resource_path_root / "conf"


@pytest.fixture()
def left_cones(inputs_path):
    """Cones images."""
    return pandora.create_dataset_from_inputs(
        {"img": str(inputs_path / "left.png"), "nodata": np.nan, "mask": None, "disp": [-60, 0]}
    )


@pytest.fixture()
def right_cones(inputs_path):
    """Cones images."""
    return pandora.create_dataset_from_inputs(
        {"img": str(inputs_path / "right.png"), "nodata": np.nan, "mask": None, "disp": [0, 60]}
    )


@pytest.fixture()
def left_rgb(inputs_path):
    """Cones images."""
    return pandora.create_dataset_from_inputs(
        {"img": str(inputs_path / "left_rgb.tif"), "nodata": np.nan, "mask": None, "disp": [-60, 0]}
    )


@pytest.fixture()
def right_rgb(inputs_path):
    """Cones images."""
    return pandora.create_dataset_from_inputs(
        {"img": str(inputs_path / "right_rgb.tif"), "nodata": np.nan, "mask": None, "disp": [0, 60]}
    )


@pytest.fixture()
def left_cones_classif(inputs_path):
    """Cones images with classification."""
    return pandora.create_dataset_from_inputs(
        {
            "img": str(inputs_path / "left.png"),
            "nodata": np.nan,
            "mask": None,
            "classif": str(inputs_path / "left_classif.tif"),
            "disp": [-60, 0],
        }
    )


@pytest.fixture()
def right_cones_classif(inputs_path):
    """Cones images with classification."""
    return pandora.create_dataset_from_inputs(
        {
            "img": str(inputs_path / "right.png"),
            "nodata": np.nan,
            "mask": None,
            "classif": str(inputs_path / "right_classif.tif"),
            "disp": [0, 60],
        }
    )


@pytest.fixture()
def left_cones_segm(inputs_path):
    """Cones images with segmentation."""
    return pandora.create_dataset_from_inputs(
        {
            "img": str(inputs_path / "left.png"),
            "nodata": np.nan,
            "mask": None,
            "segm": str(inputs_path / "left_classif.tif"),
            "disp": [-60, 0],
        }
    )


@pytest.fixture()
def right_cones_segm(inputs_path):
    """Cones images with segmentation."""
    return pandora.create_dataset_from_inputs(
        {
            "img": str(inputs_path / "right.png"),
            "nodata": np.nan,
            "mask": None,
            "segm": str(inputs_path / "right_classif.tif"),
            "disp": [0, 60],
        }
    )


@pytest.fixture()
def disp_left(outputs_path):
    return rasterio.open(outputs_path / "disp_left.tif").read(1)


@pytest.fixture()
def disp_right(outputs_path):
    return rasterio.open(outputs_path / "disp_right.tif").read(1)


@pytest.fixture()
def occlusion(outputs_path):
    return rasterio.open(outputs_path / "occl.png").read(1)


@pytest.fixture()
def disp_left_zncc(outputs_path):
    return rasterio.open(outputs_path / "disp_left_zncc.tif").read(1)


@pytest.fixture()
def disp_right_zncc(outputs_path):
    return rasterio.open(outputs_path / "disp_right_zncc.tif").read(1)


@pytest.fixture()
def left_crafted():
    """Manually computed images."""
    data = np.array(
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 4, 3], [1, 1, 1, 1, 1]),
        dtype=np.float32,
    )
    result = xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        attrs={
            "no_data_img": 0,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": None,
            "disparity_source": [-60, 0]
        },
    )
    return result


@pytest.fixture()
def right_crafted():
    """Manually computed images."""
    data = np.array(
        ([1, 1, 1, 2, 2], [1, 1, 1, 4, 2], [1, 1, 1, 4, 4], [1, 1, 1, 1, 1]),
        dtype=np.float32,
    )
    result = xr.Dataset(
        {"im": (["row", "col"], data)},
        coords={"row": np.arange(data.shape[0]), "col": np.arange(data.shape[1])},
        attrs={
            "no_data_img": 0,
            "valid_pixels": 0,
            "no_data_mask": 1,
            "crs": None,
            "transform": None,
            "disparity_source": [0, 60]
        },
    )
    return result


@pytest.fixture()
def import_plugin():
    """Import plugins in all tests without calling it."""
    pandora.import_plugin()
