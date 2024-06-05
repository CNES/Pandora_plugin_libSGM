## Data description

The **outputs** folder contains the following elements:

* **disp_left.tif, disp_right.tif, occl.png**
Our data test sample is based on the 2003 Middleburry dataset (D. Scharstein & R. Szeliski, 2003).
(D. Scharstein & R. Szeliski, 2002). Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. International journal of computer vision, 47(1-3), 7-42.
(D. Scharstein & R. Szeliski, 2003). Scharstein, D., & Szeliski, R. (2003, June). High-accuracy stereo depth maps using structured light. In 2003 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2003. Proceedings. (Vol. 1, pp. I-I). IEEE.

* **disp_left_zncc.tif, disp_right_zncc.tif**
Disparity maps computed by Pandora with configurations from inputs

* **left_disparity_3sgm.tif, right_disparity_3sgm.tif** 
Disparity maps computed by Pandora with this old configurations (no more available)

```json
{
  "input": {
    "nodata_left": NaN,
    "nodata_right": NaN,
    "left_mask": null,
    "right_mask": null,
    "left_classif": null,
    "right_classif": null,
    "left_segm": "white_band_mask.png",
    "right_segm": "white_band_mask.png",
    "disp_min_right": null,
    "disp_max_right": null,
    "img_left": "./left.png",
    "img_right": "./right.png",
    "disp_min": -60,
    "disp_max": 0
  },
  "pipeline": {
    "right_disp_map": {
      "method": "accurate"
    },
    "matching_cost": {
      "matching_cost_method": "census",
      "window_size": 5,
      "subpix": 1
    },
    "cost_volume_confidence": {
      "confidence_method": "std_intensity",
      "indicator": ""
    },
    "optimization": {
      "optimization_method": "sgm",
      "piecewise_optimization_layer": "segm",
      "overcounting": false,
      "P1": 8,
      "P2": 32,
      "p2_method": "constant",
      "penalty_method": "sgm_penalty",
      "sgm_version": "c++",
      "min_cost_paths": false
    },
    "disparity": {
      "disparity_method": "wta",
      "invalid_disparity": NaN
    },
    "refinement": {
      "refinement_method": "vfit"
    },
    "filter": {
      "filter_method": "median",
      "filter_size": 3
    },
    "validation": {
      "validation_method": "cross_checking",
      "cross_checking_threshold": 1
    },
    "filter.after.validation": {
      "filter_method": "median",
      "filter_size": 3
    }
  }
}
```