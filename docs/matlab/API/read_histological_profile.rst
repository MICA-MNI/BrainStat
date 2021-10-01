.. _matlab_read_histology_profile:

==============================
read_histology_profile
==============================

Synopsis
=============

Downloads and reads histological profiles from the BigBrain atlas (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/context/read_histology_profile.m>`_).

Usage 
=====
::

    profiles = read_histology_profile(varargin);

- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/bigbrain_data.
'template'
    - The surface template to use. Valid options are 'fsaverage', 'fsaverage5', and 'fslr32k'.
'overwrite'
    - If true, overwrites existing files, defaults to false.
