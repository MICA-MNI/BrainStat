.. _matlab_fetch_mask:

==============================
fetch_mask
==============================

Synopsis
=============

Fetch midline masks (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_mask.m>`_).

Usage 
=====
::

    mask = fetch_mask(template, varargin);
    [mask_left, mask_right] = fetch_mask(template, 'join', false, varargin);

- *template*: Cortical surface template to use. Valid values are 'civet41k' (default), 'civet164k'.
- *varargin*: Name-value pairs, see below.

Name-value pairs
================
'data_dir'
    - Directory where the data is stored. Defaults to $HOME_DIR/brainstat_data.
'overwrite'
    - If true, overwrites older files. Defaults to false.
'join'
    - If true, returns a single mask with both left and right sides, otherwise returns two separate masks, defaults to true.