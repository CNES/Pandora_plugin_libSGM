Usage
=====

Pandora configuration
---------------------

This plugins makes "sgm" optimization method avalaible on Pandora.

When this method is active, there are several sub-parameters that can be used:

+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| Name               | Description                                             | Type   | Default value | Available value                                    | Required                                                 |
+====================+=========================================================+========+===============+====================================================+==========================================================+
| penalty_estimation | Method for penalty estimation                           | string | "sgm_penalty" | "sgm_penalty","mc_cnn_penalty"                     | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| p2_method          | Method for p2 penalty estimation                        | String | "constant"    | "constant" , "negativeGradient", "inverseGradient" | No. Only available if penalty_estimation = "sgm_penalty" |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| overcounting       | overcounting correction                                 | Boolean| False         | True, False                                        | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+
| min_cost_paths     | Number of sgm paths that give the same final disparity  | Boolean| False         | True, False                                        | No                                                       |
+--------------------+---------------------------------------------------------+--------+---------------+----------------------------------------------------+----------------------------------------------------------+

There are some parameters depending on penalty_estimation choice and p2_method choice.

- penalty_estimation = "sgm_penalty" and  p2_method = "constant"

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- penalty_estimation = "sgm_penalty" and p2_method = "negativeGradient"

:math:`P2 = - \alpha \mid I(p)-I(p-r) \mid + \gamma \ ` with I for intensity on left image

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| alpha | Penalty parameter | float        | 1.0           |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| gamma | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- penalty_estimation = "sgm_penalty" and p2_method = "inverseGradient"

:math:`P2 = \frac{\alpha}{\mid I(p)-I(p-r) \mid + \beta} + \gamma \ ` with I for intensity on left image

+-------+-------------------+--------------+---------------+-----------------+----------+
| Name  | Description       | Type         | Default value | Available value | Required |
+=======+===================+==============+===============+=================+==========+
| P1    | Penalty parameter | int or float | 8             | >0              | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| P2    | Penalty parameter | int or float | 32            | P2 > P1         | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| alpha | Penalty parameter | float        | 1.0           |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| beta  | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+
| gamma | Penalty parameter | int or float | 1             |                 | No       |
+-------+-------------------+--------------+---------------+-----------------+----------+

- penalty_estimation = "mc_cnn_penalty"

.. math::
  D1 &= \mid I_{l}(p-d)-I_{l}(p-d-r) \mid \ , D2 = \mid I_{r}(p-d)-I_{r}(p-d-r) \mid \\
  P1 &= sgm_P1 \ , P2 = sgm_P2 \ if \ D1<sgm_D \ , D2<sgm_D \\
  P1 &= \frac{sgm_P1}{sgm_Q2} \ , P2 = \frac{sgm_P2}{sgm_Q2} \ if \ D1 \geq sgm_D \ , D2 \geq sgm_D \\
  P1 &= \frac{sgm_P1}{sgm_Q1} \ , P2 = \frac{sgm_P2}{sgm_Q1} \ otherwise

+------+-------------------+--------------+---------------+-----------------+----------+
| Name | Description       | Type         | Default value | Available value | Required |
+======+===================+==============+===============+=================+==========+
| P1   | Penalty parameter | int or float | 1.3           | >0              | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| P2   | Penalty parameter | int or float | 18.1          | P2 > P1         | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q1   | Penalty parameter | int or float | 4.5           |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| Q2   | Penalty parameter | int or float | 9             |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| D    | Penalty parameter | int or float | 0.13          |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+
| V    | Penalty parameter | int or float | 2.75          |                 | No       |
+------+-------------------+--------------+---------------+-----------------+----------+

.. note:: P1, P2, Q1, Q2, D, V represent sgm_P1, sgm_P2, sgm_Q1, smg_Q2, sgm_D, sgm_V respectively

Pandora's data
--------------

As a reminder, Pandora generates a cost, during the matching cost computation step. This cost volume is a
xarray.DataArray 3D float32 type, stored in a xarray.Dataset.

The plugin receives this cost volume and uses the libsgm to optimize it. Then, this optimized cost volume is returned
to Pandora.

Moreover, if "cost_min_path" option is activated, the cost volume is enriched with a new confidence_measure called
"optimization_pluginlibSGM_nbOfDisp". This 2-dimension map represents the number of sgm paths that give the same
position for minimal optimized cost at each point.








