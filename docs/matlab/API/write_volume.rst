.. _matlab_write_volume:

==============================
write_volume
==============================

Synopsis
=============

Writes volumetric data to disk (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/io/write_volume.m>`_).

Usage 
=====
::

    write_volume(filename, vol, varargin)

- *filename*: Name(s) of files. Accepted formats are .nii and .nii.gz.
- *vol*: The matrix to write.
- *varargin*: Name-value pairs, see below.

Name-value pairs
=================
Name-value pairs are used to add header information to the file.

'voxelsize'
    - Sets the voxel size in the header.
'origin'
    - Sets the origin in the header.
'datatype'
    - Sets the datatype in the header.
'description'
    - Sets the description in the header. 