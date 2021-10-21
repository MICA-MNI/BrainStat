.. _matlab_read_surface_data:

==============================
read_surface_data
==============================

Synopsis
=============

Reads data on a cortical surface (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/io/read_surface_data.m>`_).

Usage 
=====
::

    data = read_surface_data(files)

- *data*: The data read from the files. 
- *files*: Name(s) of files. Accepted formats are gifti, cifti (.dlabel.nii only), .annot, .mat (containing only one varibable), .txt, .thickness, .mgh, and .asc. 
