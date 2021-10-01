.. _matlab_FixedEffect:

==============================
FixedEffect
==============================

Synopsis
=============

Class for creating FixedEffects terms and models (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/stats/%40FixedEffect/FixedEffect.m>`_).

Usage 
=====
::

    model = FixedEffect(x, names, add_intercept);

- *model*: the FixedEffect object. 
- *x*: The vector or matrix of fixed effects. 
- *names*: Names for each effect, optional.
- *add_intercept*: Add an intercept term, optional, defaults to true.

Method Overloads
================
Several of the methods in this class have overloads that allow for combining
different FixedEffect objects. Adding two effects together (:code:`model_1 +
model_2`) results in a new model with the combined effects. Multiplying two
effects (:code:`model_1 * model_2`) together results in a new model with the
product of the effects i.e. the interaction effect. 