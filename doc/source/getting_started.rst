Getting started
===============

Overview
########

`Pandora <https://github.com/CNES/Pandora>`_ stereo matching framework is designed to provide some state of the art stereo algorithms and to add others one as plugins.
This `Pandora plugin <https://pandora.readthedocs.io/userguide/plugin.html>`_ aims to optimize the cost volume following Semi-Global Matching algorithm, defined by [Hirschmuller]_, with the `libSGM <https://github.com/CNES/Pandora_libSGM>`_  library .

.. [Hirschmuller] H. Hirschmuller. "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166


Install
#######

**pandora_plugin_libsgm** is available on Pypi and can be installed by:

.. code-block:: bash

    pip install pandora_plugin_libsgm

This command will installed required dependencies as `Pandora <https://github.com/CNES/Pandora>`_ and `libSGM <https://github.com/CNES/Pandora_libSGM>`_.

Usage
#####

Let's refer to `Pandora's readme <https://github.com/CNES/Pandora/blob/master/README.md>`_ or `online documentation <https://cnes.github.io/Pandora/index.html>`_ for further information about Pandora general functionalities.

More specifically, you can find :

- `SGM configuration file example <https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_semi_global_matching.json>`_

- `documentation about SGM theory and parameters <https://pandora.readthedocs.io/userguide/plugins/plugin_libsgm.html>`_


Related
#######


* `Pandora <https://github.com/CNES/Pandora>`_ - A stereo matching framework

* `libSGM <https://github.com/CNES/Pandora_libSGM>`_ - C++/python implementation of semi-global matching algorithm

References
##########

Please cite the following paper when using Pandora and pandora_plugin_libsgm:

*Cournet, M., Sarrazin, E., Dumas, L., Michel, J., Guinet, J., Youssefi, D., Defonte, V., Fardet, Q., 2020. Ground-truth generation and disparity estimation for optical satellite imagery. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences.*

