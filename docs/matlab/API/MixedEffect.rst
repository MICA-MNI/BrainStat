.. _matlab_MixedEffect:

==============================
MixedEffect
==============================

Synopsis
=============

Class for creating MixedEffects terms and models (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/stats/%40MixedEffect/MixedEffect.m>`_).

Usage 
=====
::

    model = MixedEffect(ran, fix, varargin);

- *ran*: A matrix containing the random effect. 
- *fix*: A matrix contianing the fixed effect, optional. 
- *varargin*: Name-value pairs, see below.
- *add_intercept*: Add an intercept term, optional, defaults to true.

Name-Value Pairs
================
'add_identity'
    - If true, adds an identity term to the model, defaults to true.
'add_intercept'
    - If true, adds an intercept term to the model, defaults to true.
'name_ran'
    - Name(s) of the random effect(s).
'name_fix'
    - Name(s) of the fixed effect(s).
'ranisvar'
    - If true, then ran is treated as if its already a term for the variance. This option is intended for developer usage only.

Method Overloads
================
Several of the methods in this class have overloads that allow for combining
different MixedEffect objects. Adding two effects together (:code:`model_1 +
model_2`) results in a new model with the combined effects. Multiplying two
effects (:code:`model_1 * model_2`) together results in a new model with the
product of the effects i.e. the interaction effect. 

