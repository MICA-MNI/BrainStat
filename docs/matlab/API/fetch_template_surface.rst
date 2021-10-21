.. _matlab_fetch_template_surface:

==============================
fetch_template_surface
==============================

Synopsis
=============

Fetches template surfaces (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_template_surface.m>`_).

Usage 
=====
::

    surface = fetch_template_surface(template, varargin);

- *surface*: The surface structure.
- *template*: Cortical surface template to use. Valid values are 'fsaverage5', 'fsaverage6', 'fsaverage', 'fslr32k', 'civet41k', 'civet164k'.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data/parcellation_data.
'layer'
    - the specific surface to load. Valid values for fsaverage are 'pial' (default), 'white', 'sphere', 'smoothwm', 'inflated'. Valid values for conte69 are: 'midthickness' (default), 'inflated', 'vinflated'. Valid values for civet are: 'mid' (default), and 'white'.