.. _matlab_SLM:

==============================
SLM
==============================

Synopsis
=============

The core object of the MATLAB BrainStat statistics module (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/stats/%40SLM/SLM.m>`_).

Usage 
=============
::

    obj = SLM(model, contrast, varargin);
    obj.fit(data);

- *obj*: the SLM object. 
- *model*: the input model, either a FixedEffect or MixedEffect object.
- *contrast*: a contrast in observations of the model. 
- *varargin*: a set of name-value pairs (see below).

Initialization
==============

A basic SLM object can initialized by simply running it with only a model and contrast
i.e. ``obj = SLM(model, contrast);``. However, several name-value pairs can
be provided to alter its behavior.  

'cluster_threshold'
    - P-value threshold or statistic threshold for defining clusters, Defaults to 0.001.
'correction'
    - A cell array containing 'rft', 'fdr', or both. If 'rft' is included, then a random field theory correction will be run. If 'fdr' is included, then a  false discovery rate correction will be run. Defaults to [].
'drlim'
    - Step of ratio of variance coefficients, in sd's. Defaults 0.1. 
'mask'
    - A logical vector containing true for vertices that should be kept during the analysis. Defaults to [].
'surf'
    - A char array containing a path to a surface, a cell/string array of the aforementioned, or a loaded surface in SurfStat format. Defaults to struct(). 
'thetalim'
    - Lower limit on variance coefficients, in sd's. Defaults 0.01
'two_tailed'
    - Whether to run one-tailed or two-tailed significance tests. Defaults to true. Note that multivariate models only support two-tailed tests.

Properties
==========

'cluster_threshold'
    - P-value threshold or statistic threshold for defining clusters, Defaults to 0.001.
'coef'
    - The coefficients of the model.
'contrast'
    - The contrast of the model.
'coord'
    - Vertex coordinates.
'correction'
    - Correction method for multiple comparisons.
'df'
    - Degrees of freedom.
'drlim'
    - Step of ratio of variance coefficients, in sd's. Defaults 0.1. 
'lat'
    - Lattice structure.
'mask'
    - A logical vector containing true for vertices that should be kept during the analysis. Defaults to [].
'model'
    - The model of the SLM.
'P'
    - The p-values corrected for random field theory.
'Q' 
    - The p-values corrected for false discovery rate.
'SSE'
    - The sum of squared errors.
'surf'
    - The surface of the model.
't'
    - The t-statistic.
'thetalim'
    - Lower limit on variance coefficients, in sd's. Defaults 0.01
'tri'
    - The surface triangles.
'two_tailed'
    - Whether to run one-tailed or two-tailed significance tests. Defaults to true. Note that multivariate models only support two-tailed tests.
'X'
    - The design matrix.

