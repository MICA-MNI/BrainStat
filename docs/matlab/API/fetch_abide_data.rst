.. _matlab_fetch_abide_data:

==============================
fetch_abide_data
==============================

Synopsis
=============

Fetch ABIDE demographic and cortical thickness data (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_abide_data.m>`_).

Usage 
=====
::

    [thickness, demographics] = fetch_abide_data(varargin);

- *thickness*: Cortical thickness on the CIVET41 template.
- *demographics*: Demographic data.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/abide_data.
sites'
    - The sites to keep subjects from. Defaults to all sites.
'keep_control'
    - If true, keeps control subjects, defaults to true.
'keep_patient'
    - If true, keeps patients, defaults to false.
'overwrite'
    - If true, overwrites older files. Defaults to false.