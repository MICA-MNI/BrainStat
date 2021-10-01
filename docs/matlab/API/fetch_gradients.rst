.. _matlab_fetch_gradients:

==============================
fetch_gradients
==============================

Synopsis
=============

Fetch precomputed gradients (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_gradients.m>`_).

Usage 
=====
::

    gradients = fetch_gradients(template, name, varargin);

- *template*: Cortical surface template to use. Valid values are 'fsaverage5' (default), 'fsaverage', and 'fslr32k'
- *name*: Name of the gradients. Currently only 'margulies2016' is accepted. Defaults to 'margulies2016'
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/gradient_data.
'overwrite'
    - If true, overwrites older files. Defaults to false.