.. _volume_viewer_matlab:

==================
volume_viewer
==================

------------------
Synopsis
------------------

Plots data in a volume (`source code
<https://github.com/MICA-MNI/BrainStat/blob/master/matlab/visualization/volume_viewer/%40volume_viewer/volume_viewer.m>`_).


------------------
Usage
------------------

.. code-block:: matlab

   obj = volume_viewer(structural,overlay,varargin);

- *structural*: volume of a structural image. 
- *overlay*: (Optional argument) an overlay to plot over the image volume, must have identical dimensions. 
- *varargin*: Name-Value Pairs (see below).
- *obj*: an object used for the manipulation of the volume viewer.

------------------
Description
------------------

BrainStat's volume viewer allows for the viewing of statistcal maps in volume space.
The volume viewer allows for scrolling and clicking through the volume. Note
that slice orientation is currently hardcoded and cannot be modified. 

Name-Value Pairs
^^^^^^^^^^^^^^^^^
- *remove_zero*: Does not display zeros in the overlay (default: true). 
- *threshold_lower*: Does not display values of the overlay below the minimum of the colorbar (default: false).
- *threshold_upper*: Does not display values of the overlay above the maximum of the colorbar (default: false).


Object Properties
^^^^^^^^^^^^^^^^^^^
- *slices*: Contains the curently displayed slices. 
- *threshold_lower*: See name-value pair with identical name.
- *threshold_upper*: See name-value pair with identical name.
- *remove_zero*: See name-value pair with identical name.
- *image*: Contains the input image (not modifiable).
- *overlay*: Contains the input overlay (not modifiable).
- *handles* Contains handles to the graphics objects (not modifiable).

Object Methods
^^^^^^^^^^^^^^^
- obj.colorlimits(limits,image): Sets the color limits of the structural (image="image") or overlay (image="overlay") to limits. 
- obj.colormap(cmap,image): Sets the color map of the structural (image="image") or overlay (image="overlay") to cmap. 
