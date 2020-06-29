Upper limit and invalid value
=============================


Upper limit of the cost volume
------------------------------

We aim at demonstrating that cost volume is limited by an upper limit

As a reminder, SGM equation is :

.. math::

    & L(p,d) = C(p,d) + min(L_{r}(p-r,d),L_{r}(p-r,d+1)+P1, L_{r}(p-r,d-1)+ P1 , min(L_{r}(p-r, d_{i}) + P2) - min(L_{r}(p-r, d_{k})) \\
    & P2 > P1 > 0 \ , \ 0 < C(p,d) < C_{max} \\
    & k \in [0,D] , i \in [0,d-1[ \cup ]d+1,D] \\

Two cases can be distinguished

- First one is :math:`min(L_{r}(p-r, d_{i}) = min(L_{r}(p-r, d_{k})`

Let's denote :math:`m_{Prox} = min(L_{r}(p-r,d),L_{r}(p-r,d+1)+P1, L_{r}(p-r,d-1)+ P1)`

:math:`If \ min(m_{Prox}, min(L_{r}(p-r, d_{i}) + P2) = min(L_{r}(p-r, d_{i}) + P2`

.. math::
  & \Rightarrow L(p,d) = C(p,d) + P2 \\
  & \Rightarrow L(p,d) \leq C_{max} + P2

:math:`Else \ min(m_{Prox}, min(L_{r}(p-r, d_{i}) + P2) = m_{Prox}`

.. math::
  & \Rightarrow a <  min(L_{r}(p-r, d_{i}) + P2 \\
  & \Rightarrow a - min(L_{r}(p-r, d_{i}) < P2 \\
  & \Rightarrow L(p,d) = C(p,d) + a - min(L_{r}(p-r, d_{i}) < C(p,d) + P2 \\
  & \Rightarrow L(p,d) < C_{max} + P2

- The second one is :math:`min(L_{r}(p-r, d_{i}) \neq min(L_{r}(p-r, d_{k}) , k \in [d-1,d,d+1]`

Let's denote :math:`m = min(L_{r}(p-r,d),L_{r}(p-r,d+1)+P1, L_{r}(p-r,d-1)+ P1, min(L_{r}(p-r, d_{i}) + P2)`

If :math:`k=d`

.. math::
  & \Rightarrow m = L_{r}(p-r,d) \\
  & \Rightarrow L(p,d) = C(p,d)

Else :math:`k \in {d-1,d+1}`

.. math::
  & If \ m = min( L_{r}(p-r,d+1)+P1, L_{r}(p-r,d-1) + P1 ) \\
  & \Rightarrow  L(p,d) = C(p,d) + P1

.. math::
  & Else \ m = L_{r}(p-r,d) \\
  & \Rightarrow L(p,d) = C(p,d) + L_{r}(p-r,d) - L_{r}(p-r,d+1) \\
  & or \ L(p,d) = C(p,d) + L_{r}(p-r,d) - L_{r}(p-r,d-1) \\
  & \Rightarrow L(p,d) < C(p,d) + P1 < C_{max} + P2


The upper limit of the cost volume is :math:`C_{max} + P2`.

Invalid points management
----------------------------

The cost volume, as an input of libSGM, can contain invalid points. We consider them as invalid for two possible reasons:

 1. Disparity range is not wide enough so homologuous point cannot be identified.
 2. Point can be considered as invalid following an input mask given by user.

These invalid points must be skipped by the SGM optimisation. And at the end of the algorithm, these points must stay invalid.

Following the previous demonstration about upper limit equal to :math:`C_{max} + P2`, value of invalid points is set to :math:`C_{max} + P2 + 1`.

During SGM optimization, when an invalid point is encountered, its value is left at :math:`C_{max} + P2 + 1`.

Thanks to this, we make sure that no invalid point can be considered during SGM computation. To prove it, we have to distinguish two cases:

1. For a given point p, only a few disparities are invalid.

No any problem. The fact that there are valid points (p,d), whose value is known to be less than :math:`C_{max} + P2`,
assures that no invalid point will be considered during computation.

2. For a given point p, all disparities are invalid.

:math:`\forall d \in [0,D], L_{r}(p-r,d) = C_{max} + P2 + 1`

.. math::
  & \Rightarrow min(L_{r}(p-r,d),L_{r}(p-r,d+1)+P1, L_{r}(p-r,d-1)+ P1 , min(L_{r}(p-r, d_{i}) + P2) = C_{max} + P2 + 1 \\
  & and \ min(L_{r}(p-r, d_{k})) = C_{max} + P2 + 1 \\
  & \Rightarrow L(p,d) = C(p,d)

Since version 1.B, penalties may no longer be constant. The previous demonstration was made under the assumption that P2 is a constant parameter.
When P2 is not, we can easily demonstrate that upper limit is equal to :math:`C_{max} + P2_{max}`.
In the same way as the constant case, invalid point must be set to a value greater or equal to :math:`C_{max}+P2_{max}+1`.

So, we must calculate :math:`P2_{max}` for each P2 estimation methods.

- Inverse gradient

.. math::
    & P2 = - \alpha \mid I(p)-I(p-r) \mid + \gamma \\
    & 0 \leq \mid I(p)-I(p-r) \mid \leq I_{max} \ and\  \alpha , \beta , \gamma > 0 \\
    & \Rightarrow -I_{max} \leq - \mid I(p)-I(p-r) \mid \leq 0 \\
    & \Rightarrow -\alpha I_{max} + \gamma \leq - \alpha \mid I(p)-I(p-r) \mid + \gamma \leq \gamma \\
    & \Rightarrow P2_{max} = \gamma \\
    & \Rightarrow InvalidValue = C_{max} + \gamma + 1

- Negative gradient

.. math::
    & P2 = - \alpha \mid I(p)-I(p-r) \mid + \gamma \\
    & 0 \leq \mid I(p)-I(p-r) \mid \leq I_{max} \ and\  \alpha , \beta , \gamma > 0 \\
    & \Rightarrow \beta \leq \mid I(p)-I(p-r) \mid + \beta \leq I_{max} + \beta \\
    & \Rightarrow \frac{\alpha}{I_{max}+\beta} + \gamma \leq \frac{\alpha}{\mid I(p)-I(p-r) \mid +\beta} + \gamma \leq \frac{\alpha}{\beta} + \gamma \\
    & \Rightarrow P2_{max} = \frac{\alpha}{\beta} + \gamma \\
    & \Rightarrow InvalidValue = C_{max} + \frac{\alpha}{\beta} + \gamma + 1

- mc-cnn penalty

.. math::
  D1 &= \mid I_{l}(p-d)-I_{l}(p-d-r) \mid \ , D2 = \mid I_{r}(p-d)-I_{r}(p-d-r) \mid \\
  P2 &= sgm_P2 \ if \ D1<sgm_D \ , D2<sgm_D \\
  P2 &= \frac{sgm_P2}{sgm_Q2} \ if \ D1 \geq sgm_D \ , D2 \geq sgm_D \\
  P2 &= \frac{sgm_P2}{sgm_Q1} \ otherwise

To determine the P2 to use to compute invalid value, among the 3 possible P2 values, we take the maximum one. Indeed the invalid value must have the highest
possible value.

.. math::
    & \Rightarrow InvalidValue = C_{max} + max(sgm_P2, \frac{sgm_P2}{sgm_Q2}, \frac{sgm_P2}{sgm_Q1})+ 1
