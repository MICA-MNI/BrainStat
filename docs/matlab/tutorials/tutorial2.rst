.. _matlab_context:

Tutorial 2: Context Module
==========================

In this tutorial you'll learn how to use the context module. The context module
allows for the contextualization of your statistics maps with genetics, meta-analytic
and histological datasets. (Note: Currently only the genetics dataset is supported).
Before starting this tutorial, please make sure that you have installed all dependencies
for BrainStat MATLAB (see :ref:`install_page`). 


Genetics
--------
For genetics contextualization we use `abagen
<https://abagen.readthedocs.io/en/stable/index.html>`_  to connect to the `Allen
Human Brain Atlas <https://human.brain-map.org/>`_. First, make sure that your 
Python BrainStat is installed and correctly linked to your MATLAB. If it's not
then please consult the installation page before continuing with this tuturial.
To check your Python environment run the following in MATLAB:

.. code-block:: MATLAB

    env = pyenv; 
    env.ExecutionMode % This should return InProcess
    context_utils.py_test_environment('brainstat', 'pyembree') % Should return true.

For our example data, we'll simply load data that came with the Python package
"nilearn" that was installed together with brainstat. If the following code
looks a tad complicated, don't worry about it. What's important is that you
know that variables :code:`pial_left` and :code:`white_left` are paths to pial and white
surfaces (e.g. derived from Freesurfer), :code:`desitreux` contains parcellation
labels for each vertex in a double array or a path to parcellation gifti/csv
file, and :code:`mni152_file` is the path to a volume in MNI152 space. We grab this
data as follows:

.. code-block:: MATLAB

    % Get the fsaverage5 left white and pial surface.
    fsaverage5 = py.nilearn.datasets.fetch_surf_fsaverage();
    pial_left = char(fsaverage5.get('pial_left'));
    white_left = char(fsaverage5.get('white_left'));

    % Get the parcellation from Python
    desitreux = py.nilearn.datasets.fetch_atlas_surf_destrieux();
    desitreux_np_array = desitreux.get('map_left');
    desitreux_cell = cell(desitreux_np_array.tolist());
    desitreux = cellfun(@double, desitreux_cell);

    % Load the MNI152 template
    mni152_obj = py.nilearn.datasets.load_mni152_template();
    mni152_file = char(mni152_obj.dataobj.file_like);

With your example data loaded, we can now get the genetic expression of all
desitreux regions. Please be aware that to compute this we'll have to download
the Allen Human Brain Atlas dataset (approximately 4GB), and that the entire
process may take up to half an hour. By default the dataset will be downloaded
to your BrainStat directory, but you can modify this by altering the :code:`data_dir`
name-value pair.

.. code-block:: MATLAB

    data_dir = '/path/to/data/directory';
    [expression, gene_names] = surface_genetic_expression(pial_left, ...
        white_left, desitreux, mni152_file, 'data_dir', data_dir);

The variable :code:`expression` now contains the expression values in a region-by-gene
matrix variable :code:`gene_names` contains the name of each gene. Note that if no data
was available for a region, then this results in a NaN row. 

In many cases, you may want to get the genetic expression for both hemispheres
simultaneously. In that case simply provide both hemispheres in a cell array in
variables :code:`pial_left`, :code:`white_left`, and :code:`desitreux`. Make sure you provide them
in the same order in all variables!

This tutorial showcases the basic usage of the genetics module. For a more
complete tutorial, including the meaning of all name-value pairs, see the
`abagen documentation <https://abagen.readthedocs.io/en/stable/index.html>`_.