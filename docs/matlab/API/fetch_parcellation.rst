.. _matlab_fetch_parcellation:

==============================
fetch_parcellation
==============================

Synopsis
=============

Fetches cortical parcellations (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_parcellation.m>`_).

Usage 
=====
::

    parcellation = fetch_parcellation(template, atlas, n_regions, varargin);

- *template*: Cortical surface template to use. Valid values are 'fsaverage5', 'fsaverage6', 'fsaverage', 'fslr32k'.
- *atlas*: Name of a surface parcellation. Valid names are: 'schaefer', 'cammoun', 'glasser', 'yeo'.
- *n_regions*: Number of regions in the parcellation, see function help for details on the number of regions available for each atlas.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/parcellation_data.
'seven_networks'
    - Only used for the Schaefer atlas. If true, uses the 7 network Yeo sub-parcellation, otherwise uses the 17 network Yeo sub-parcellation, defaults to true.
