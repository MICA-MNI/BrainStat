.. _install_page:

Installation Guide
==============================

BrainStat is available in Python and MATLAB.


Python installation
-------------------

BrainStat is compatible with Python 3.6-3.8. Compatibility with 3.9 is in the
works.


Installation
^^^^^^^^^^^^

BrainStat can be installed using ``pip`` (NOTE: This doesn't work yet!): ::

    pip install brainstat


Alternatively, you can install the package from Github as follows: ::

    git clone https://github.com/MICA-LAB/BrainStat.git
    cd BrainStat
    python setup.py install


MATLAB installation
-------------------

This toolbox has been tested with MATLAB versions R2019b and newer. 

To install the MATLAB toolbox simply `download
<https://github.com/MICA-LAB/BrainStat/releases>`_ and unzip the GitHub toolbox and run
the following in MATLAB: ::

    addpath(genpath('/path/to/BrainSpace/matlab/'))

If you want to load BrainStat every time you start MATLAB, type ``edit
startup`` and append the above line to the end of this file. 
    
If you wish to open gifti files you will also need to install the `gifti library
<https://www.artefact.tk/software/matlab/gifti/>`_.