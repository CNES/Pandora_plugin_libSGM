Quick overview
==============

Pandora plugin to optimize the cost volume following SGM algorithm [1]_ with the libSGM library.

One can implement their own penalty estimation methods, corresponding to P1 and P2 parameters of SGM equation.
Some are already avalaible and computed by the plugin_libsgm:

* Methods depending on intensity gradient of the left image [2]_.
* Method depending on intensity gradient of left and right image [3]_.

.. [1] H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, Feb. 2008. doi: 10.1109/TPAMI.2007.1166
.. [2] Banz, C. & Pirsch, P. & Blume, Holger. (2012). EVALUATION OF PENALTY FUNCTIONS FOR SEMI-GLOBAL MATCHING COST AGGREGATION. ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences. XXXIX-B3. 1-6. 10.5194/isprsarchives-XXXIX-B3-1-2012.
.. [3] Zbontar, Jure and Yann LeCun. “Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches.” ArXiv abs/1510.05970 (2016): n. pag.