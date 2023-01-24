<h1 align="center"> Pandora plugin libSGM </h1>

<h4 align="center">Semi-Global Matching algorithm plugin for <a href="https://github.com/CNES/Pandora"><img align="center" src="https://raw.githubusercontent.com/CNES/Pandora/master/doc/sources/Images/logo/logo_typo.svg?inline=false" width="64" height="64"/></a>  .</h4>

<p align="center">
  <a href="https://github.com/CNES/Pandora_plugin_libSGM/actions"><img src="https://github.com/CNES/Pandora_plugin_libSGM/actions/workflows/pandora_plugin_libsgm_ci.yml/badge.svg?branch=master"></a>
<a href="https://codecov.io/gh/CNES/Pandora_plugin_libSGM"> <img src="https://codecov.io/gh/CNES/Pandora_plugin_libSGM/branch/master/graph/badge.svg?token=O22Y1OF63L"></a>
  <a href="https://opensource.org/licenses/Apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
    <a href="#usage">Usage</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>

## Overview

[Pandora](https://github.com/CNES/Pandora) stereo matching framework is designed to provide some state of the art stereo algorithms and to add others one as plugins.  
This [Pandora plugin](https://pandora.readthedocs.io/en/stable/userguide/plugin.html) aims to optimize the cost volume following Semi-Global Matching algorithm, defined by [[Hirschmuller]](#Hirschmuller), with the [libSGM](https://github.com/CNES/Pandora_libSGM)  library .

## Install

**pandora_plugin_libsgm** is available on Pypi and can be installed by:

```bash
pip install numpy
pip install pandora_plugin_libsgm
```

This command will installed required dependencies as [Pandora](https://github.com/CNES/Pandora) and [libSGM](https://github.com/CNES/Pandora_libSGM).

## Usage

Let's refer to [Pandora's readme](https://github.com/CNES/Pandora/blob/master/README.md) or [online documentation](https://pandora.readthedocs.io/?badge=latest) for further information about Pandora general functionalities. 

More specifically, you can find :
- [SGM configuration file example](https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_semi_global_matching.json)
- [documentation about SGM theory and parameters](https://pandora.readthedocs.io/en/stable/userguide/plugins/plugin_libsgm.html)


## Related

[Pandora](https://github.com/CNES/Pandora) - A stereo matching framework  
[libSGM](https://github.com/CNES/Pandora_libSGM) - C++/python implementation of semi-global matching algorithm  

## References

Please cite the following paper when using Pandora and pandora_plugin_libsgm:   
*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*

<a id="Hirschmuller">[Hirschmuller]</a> 
*H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166*
