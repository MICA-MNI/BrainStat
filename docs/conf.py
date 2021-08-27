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

sys.path.insert(0, os.path.abspath(".."))

# Run a custom scraper instead of using brainspace.plotting._get_sg_image_scraper
from brainspace.plotting.base import Plotter
from brainspace.vtk_interface.wrappers import BSScalarBarActor

def _get_sg_image_scraper():
    return Scraper()
class Scraper(object):

    def __call__(self, block, block_vars, gallery_conf):
        """
        Called by sphinx-gallery to save the figures generated after running
        example code.
        """
        try:
            from sphinx_gallery.scrapers import figure_rst
        except ImportError:
            raise ImportError('You must install `sphinx_gallery`')
        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        for k, p in Plotter.DICT_PLOTTERS.items():
            fname = next(image_path_iterator)

            for _, lren in p.renderers.items():
                for r in lren:
                    for i in range(r.actors2D.n_items):
                        a = r.actors2D[i]
                        if not isinstance(a, BSScalarBarActor):
                            continue
                        a.labelTextProperty.fontsize = a.labelTextProperty.fontsize * 3

            p.screenshot(fname, scale=1)
            # p.screenshot(fname)
            image_names.append(fname)

        Plotter.close_all()  # close and clear all plotters
        return figure_rst(image_names, gallery_conf["src_dir"])


# -- Project information -----------------------------------------------------

project = "BrainStat"
copyright = "2021, MICA Lab, CNG Lab"
author = "MICA Lab, CNG Lab"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Automatic documentation
    "sphinx.ext.autosummary",  # Automatic documentation
    "sphinx.ext.viewcode",  # Add source code link
    "sphinx.ext.napoleon",  # Parses docstrings
    "sphinx.ext.intersphinx",  # Links code to other packages
    "sphinx.ext.doctest",  # Runs doctests
    "sphinxcontrib.matlab",  # MATLAB plugin
    "sphinx_gallery.gen_gallery",  # Example gallery
]

from sphinx_gallery.sorting import FileNameSortKey


sphinx_gallery_conf = {
    "examples_dirs": "python/tutorials",
    "gallery_dirs": "python/generated_tutorials",
    'thumbnail_size': (250, 250),
    'image_scrapers': ('matplotlib', _get_sg_image_scraper()),
    'within_subsection_order': FileNameSortKey,
    'download_all_examples': False,
    'remove_config_comments': True,
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
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "nilearn": ("https://nilearn.github.io/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "brainstat": ("https://brainstat.readthedocs.io/en/latest", None),
    "brainspace": ("https://brainspace.readthedocs.io/en/latest", None),
}

# Autosummary settings
autosummary_generate = True

# MATLAB documentation settings
this_dir = os.path.dirname(os.path.abspath(__file__))
matlab_src_dir = os.path.abspath(os.path.join(this_dir, ".."))
matlab_keep_package_prefix = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
