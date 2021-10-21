.. _matlab_read_volume:

==============================
read_volume
==============================

Synopsis
=============

Reads data in a volume file (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/io/read_volume.m>`_).

Usage 
=====
::

    [volume, header] = read_volume(file)

- *volume*: The volumetric data.
- *header*: The header data. 
- *file*: Name of file. Must be a .nii or nii.gz file.