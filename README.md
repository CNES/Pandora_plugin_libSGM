# Plugin LibSgm

Pandora plugin to optimize the cost volume with the LigSGM library

## Installation

**Non-developper mode**

This procedure allows you to install the plugin_libsgm, pandora and libsgm, without prior cloning them. 
Note that sources will not be accessible with this procedure.

To install it, follow the steps:

```sh
u@m $ python -m venv myEnv
u@m $ source myEnv/bin/activate
(myEnv) u@m $ pip install --upgrade pip
(myEnv) u@m $ pip install numpy
(myEnv) u@m $ pip install pandora_plugin_libsgm
```

**Developper mode**

This procedure allows you to install the plugin_libsgm, pandora, libsgm and have access to the sources.

To install it, follow the steps:

- Initializing the environment

```sh
u@m $ python -m venv myEnv
u@m $ source myEnv/bin/activate
(myEnv) u@m $ pip install --upgrade pip
(myEnv) u@m $ pip install numpy
```

- Pandora installation

```sh
(myEnv) u@m $ git clone https://github.com/CNES/Pandora_pandora.git
(myEnv) u@m $ cd Pandora_pandora
(myEnv) u@m $ pip install .
```

- LibSGM installation

```sh
(myEnv) u@m $ git clone https://github.com/CNES/Pandora_libsgm.git
(myEnv) u@m $ cd Pandora_libsgm
(myEnv) u@m $ pip install .
```

- Plugin installation

```sh
(myEnv) u@m $ git clone https://github.com/CNES/Pandora_plugin_libsgm.git
(myEnv) u@m $ cd Pandora_libsgm
(myEnv) u@m $ pip install .
```

## Documentation

**Build documentation**
Make sure  latex and dvipng is already installed

```
pip install sphinx-rtd-theme
python setup.py build_sphinx
```
Documentation is built in plugin_libsgm/doc/build/html 

Documentation is available from the [pandora](https://github.com/CNES/Pandora_pandora) and 
[libsgm](https://github.com/CNES/Pandora_libsgm) repositories.

**How to find P2 penalty parameter:**
For Census measure, the P2 range determined is [15, 120]. For a window_size of 5x5, its is Cmax=25.

p2_min_census, p2_max_census and cmax_census are used to determined the P2 range of other measures
thanks to P2_census / Cmax_census ratio.

Thus to determine P2 range of a new measure:
p2_min_measure = cmax_measure * (p2_min_census / cmax_census)
p2_max_measure = cmax_measure * (p2_max_census / cmax_census)


## Usage

**Non-developper mode**

Run pandora : 

    pandora config.json output_dir

with the config.json file : 

```json
{
  "input" : {
    "img_ref" : "PATH/TO/img_ref.tif",
    "img_sec" : "PATH/TO/img_sec.tif",
    "disp_min" : -100,
    "disp_max" : 100,
    "ref_mask" : "PATH/TO/ref_mask.tif",
    "sec_mask" : "PATH/TO/sec_mask.tif"
  },
  "stereo" : {
    "stereo_method": "census",
    "window_size": 5,
    "subpix": 1
  },
  "optimization" : {
    "optimization_method": "sgm",
    "P1": 8,
    "P2": 32,
    "penalty_method": "sgm_penalty"
  },
  "refinement": {
    "refinement_method": "vfit"
  },
 "filter" : {
   "filter_method": "median",
   "filter_size": 3
  },
  "validation" : {
    "validation_method": "cross_checking",
    "cross_checking_threshold": 1.,
    "right_left_mode": "accurate"
  }
}
```

**Developper mode**

Run pandora, with the configuration file of the plugin_libsgm:

    pandora plugin_libsgm/conf/sgm.json output_dir


## Notes

For tests, we use images coming from 2003 Middleburry dataset 
(D. Scharstein and R. Szeliski. High-accuracy stereo depth maps using structured light.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2003), 
volume 1, pages 195-202, Madison, WI, June 2003.)


## References

If you use this CNES software, please cite the following paper: 

Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. 
Ground-truth generation and disparity estimation for optical satellite imagery.
ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.

