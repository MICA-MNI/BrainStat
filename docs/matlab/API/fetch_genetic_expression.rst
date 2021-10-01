.. _matlab_fetch_genetic_expression:

==============================
fetch_genetic_expression
==============================

Synopsis
=============

Downloads and reads genetic expression profiles (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/context/fetch_genetic_expression.m>`_).

Usage 
=====
::

    profiles = fetch_genetic_expression(atlas, n_regions, varargin);

- *profiles*: Histological profiles. 
- *atlas*: Name of a surface parcellation. Valid names are: 'schaefer', 'cammoun', 'glasser'.
- *n_regions*: Number of regions in the parcellation, see function help for details on the number of regions available for each atlas.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/genetic_data.
'seven_networks'
    - Only used for the Schaefer atlas. If true, uses the 7 network Yeo sub-parcellation, otherwise uses the 17 network Yeo sub-parcellation, defaults to true.
'overwrite'
    - If true, overwrites existing files, defaults to false.
'verbose'
    - If true, provides verbose output, defaults to true. 
