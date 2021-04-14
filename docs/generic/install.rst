.. _install_page:

Installation Guide
==============================

BrainStat is available in Python and MATLAB.


Python installation
-------------------

BrainStat is compatible with Python 3.6-3.8. Compatibility with 3.9 is in the
works.


Whilst development is still ongoing, the latest version of BrainStat can be
installed using ``pip``: ::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple brainstat

Python Dependencies
+++++++++++++++++++++++

If you want to use the context module, you'll also have to download and install
the package pyembree. This package is only available through conda-forge: ::

    conda install -c conda-forge pyembree


MATLAB installation
-------------------

This toolbox is compatible with MATLAB versions R2019b and newer. 

We recommend installing the toolbox through the Mathworks `FileExchange
<https://www.mathworks.com/matlabcentral/fileexchange/89827-brainstat>`_. Simply
download the file as a toolbox and open the .mltbx file in MATLAB.
Alternatively, you can install the same .mltbx file from our `GitHub Releases
<https://github.com/MICA-MNI/BrainStat/releases>`_.

If you don't want to install BrainStat as a MATLAB Toolbox, you can also simply
`download <https://github.com/MICA-MNI/BrainStat>`_ the repository and and run
the following in MATLAB: ::

    addpath(genpath('/path/to/BrainSpace/brainstat_matlab/'))

If you want to load BrainStat every time you start MATLAB, type ``edit
startup`` and append the above line to the end of this file. 
  
MATLAB Dependencies
+++++++++++++++++++++++

If you wish to open gifti files you will also need to install the `gifti library
<https://www.artefact.tk/software/matlab/gifti/>`_.

If you wish to use the context module, then you will have to install a Python
environment, install Python BrainStat (see instructions above), and link your
MATLAB to this Python environment. To do this, first `install miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_ on your computer. Next,
create a Python environment of a Python version `compatible with your MATLAB
version
<https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf>`_
by running the following (replace "3.7" with a compatible Python version): ::

    conda create -c conda-forge -n MY_ENVIRONMENT_NAME python=3.7 pyembree
    conda activate MY_ENVIRONMENT_NAME
    pip install -i https://test.pypi.org/simple/ brainstat

Now that you have your Python environment set up, find the Python executable by
starting this new environment's Python with `python3`, then run: ::

    import sys
    print(sys.executable)

Next, open up a new MATLAB session and run the following to set your default
Python environment. ::

    pyenv('Version', PATH_TO_PYTHON_EXECUTABLE, ...
          'ExecutionMode', 'InProcess');

To test whether you've installed BrainStat Python correctly, try running the
following in MATLAB: ::

    ~isa(py.importlib.util.find_spec("brainstat"),'py.NoneType')

This will return true if Python BrainStat was found.

