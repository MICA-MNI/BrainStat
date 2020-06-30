.. _theory_page:

(Some) Theory
==============================

The following expandable section covers some of the theoretical and mathematical underpinnings of BrainStat. 


Mass-univariate linear modelling 
-------------------

BrainStat implements element-wise univariate and multivariate linear models, similar to one its predecessor `SurfStat <http://www.math.mcgill.ca/keith/surfstat/>`_ 

Linear models describe a continuous response variable as a function of one or more predictors (which can be continuous or factorial). An example of such a model is  

	Y = b0 + b1*x1 + b2*x2 + e 
	
where the b_i are the parameter estimates, x_i are the variables and e represents the error term, which is assumed to be iid. BrainStat has adopted the straightforward formula nomenclature from SurfStat, in which the above model could be specified as 

	Model = 1 + term(x1) + term(x2) 
	
followed by simple model fitting 
	
	slm = BrainStatLinMod....

Within a specified model, one can then interrogate specific contrasts, i.e. effects of variables (or variable combinations) specified in the model. The respective code for this will be. 

	slm = BrainStatT(slm, contrast) 

Where contrast could be something like x1, -x1 from the above model in the case of continuous predictor variables, such as age.  One could also specify the contrast as x1.level1 - x1.level2 should x be a factorial variable. An example could be that x is a variable indicating sex, then the 


Mixed effects models 
-------------------
BrainStat also incorporates element-wise linear mixed effects models, again leveraging functionality from `SurfStat <http://www.math.mcgill.ca/keith/surfstat/>`_ 

Mixed models allow for the 
Correction for multiple comparisons  
-------------------
Mass univariate analyses implements mass univariate linear models and mixed-effects models, similar to one of its predecessors SurfStat <http://math.mcgill.ca/keith/surfstat/>. 


Multivariate Techniques  
-------------------

BrainStat can be installed using ``pip``: ::

    pip install brainstat


Alternatively, you can install the package from Github as follows: ::

    git clone https://github.com/MICA-MNI/BrainStat.git
    cd BrainStat
    python setup.py install



MATLAB installation
-------------------

This toolbox has been tested with MATLAB versions R2018b, although we expect it
to work with versions R2017a and newer. It will definitely throw errors with
versions R2016b and older. Operating systems used during testing were OSX Mojave (10.14.6)
and Linux Xenial Xerus (16.04.6).

To install the MATLAB toolbox simply `download
<https://github.com/MICA-MNI/BrainStat/releases>`_ and unzip the GitHub toolbox and run
the following in MATLAB: ::

    addpath(genpath('/path/to/BrainSpace/matlab/'))

If you want to load BrainSpace every time you start MATLAB, type ``edit
startup`` and append the above line to the end of this file. 

You can move the MATLAB directory to other locations. However, the example data
loader functions used in our tutorials require the MATLAB and shared directories
to both be in the same directory. 
    
If you wish to open gifti files you will also need to install the `gifti library
<https://www.artefact.tk/software/matlab/gifti/>`_.
