# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'BrainStat'
copyright = '2021, MICA Lab, CNG Lab'
author = 'MICA Lab, CNG Lab'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', # Automatic documentation
    'sphinx.ext.autosummary', # Automatic documentation
    'sphinx.ext.viewcode', # Add source code link
    'sphinx.ext.napoleon', # Parses docstrings
    'sphinx.ext.intersphinx', # Links code to other packages
    'sphinx.ext.doctest', # Runs doctests
    'sphinxcontrib.matlab', # MATLAB plugin
    'sphinx_gallery.gen_gallery', # Example gallery
]

# Sphinx gallery settings.
sphinx_gallery_conf = {
    'examples_dirs': 'python/tutorials',
    'gallery_dirs': 'python/generated_tutorials'
}

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'nilearn': ('https://nilearn.github.io/', None),
    'nibabel': ('https://nipy.org/nibabel/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'brainstat': ('https://brainstat.readthedocs.io/en/latest', None),
    'brainspace': ('https://brainspace.readthedocs.io/en/latest', None),
}

# Autosummary settings
autosummary_generate = True

# MATLAB documentation settings
this_dir = os.path.dirname(os.path.abspath(__file__))
matlab_src_dir = os.path.abspath(os.path.join(this_dir, '..'))
matlab_keep_package_prefix = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
