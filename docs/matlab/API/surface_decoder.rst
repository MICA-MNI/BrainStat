.. _matlab_surface_decoder:

==============================
surface_decoder
==============================

Synopsis
=============

Decodes input data using meta-analytic maps derived from Nimare and Neurosynth (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/context/surface_decoder.m>`_).

Usage 
=====
::

    [pearsons_r_sort, feature_names_sort] = surface_decoder(stat_data,varargin);

- *pearsons_r_sort*: Correlations between the input map and each feature.
- *feature_names_sort*: Names of the features.
- *stat_data*: Data to be compared to the Neurosynth database.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'template'
    - The template surface to use, either 'fsaverage5' or 'fsaverage', or 'fslr32k', defaults to 'fsaverage5'.
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/neurosynth_data.
'ascending'
    - If true sort output in ascending order, otherwise descending, defaults to false.