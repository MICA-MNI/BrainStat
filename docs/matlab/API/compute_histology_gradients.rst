.. _matlab_compute_histology_gradients:

==============================
compute_histology_gradients
==============================

Synopsis
=============

Computes microstructural profile covariance from histological profiles (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/context/compute_histology_gradients.m>`_).

Usage 
=====
::

    gm = compute_histology_gradients(mpc, varargin);

- *gm*: GradientMaps object, see the BrainSpace documentation for details. 
- *mpc*: Microstructural profile covariance, see :ref:`matlab_compute_mpc`
- *varargin*: Accepts all options accepted by the GradientMaps object. 
