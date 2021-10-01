.. _matlab_compute_mpc:

==============================
compute_mpc
==============================

Synopsis
=============

Computes microstructural profile covariance from histological profiels (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/context/compute_mpc.m>`_).

Usage 
=====
::

    mpc = compute_mpc(profile, labels));

- *profiles*: Histological profiles, see :ref:`matlab_read_histology_profile`. 
- *labels*: Parcellation scheme on the same template as the profiles. 
