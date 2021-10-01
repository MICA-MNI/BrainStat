.. _matlab_fetch_yeo_networks_metadata:

==============================
fetch_yeo_networks_metadata
==============================

Synopsis
=============

Fetches template surfaces (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/brainstat_matlab/datasets/fetch_yeo_networks_metadata.m>`_).

Usage 
=====
::

    [network_names, colormap] = fetch_yeo_networks_metadata(n_regions);

- *network_names*: Names of the networks in the same order as those fetched with :ref:`fetch_parcellation`.
- *colormap*: Classical colormap associated with the Yeo networks.
- *n_regions*: Number of Yeo networks, either 7 or 17. 
