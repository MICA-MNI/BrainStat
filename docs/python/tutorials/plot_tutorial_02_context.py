"""
Tutorial 02: Context Decoding
=========================================

In this tutorial you will learn about the context decoding tools included with
BrainStat. The context decoding module consists of three parts: genetic
decoding, meta-analytic decoding and histological comparisons. First, we'll
consider how to run the genetic decoding analysis. 


Genetics
--------

For genetic decoding we use the Allen Human Brain Atlas through the abagen
toolbox. Note that abagen only accepts parcellated data. Here is a minimal
example of how we use abagen to get the genetic expression of the regions of the
Destrieux atlas. Please note that downloading the dataset and running this
analysis can take several minutes. As such, we will not run the analysis here.
"""

from brainstat.context.genetics import surface_genetic_expression
from nilearn import datasets
import numpy as np

run_analysis = False  # Too resource intensive to run on ReadTheDocs

destrieux = datasets.fetch_atlas_surf_destrieux()
labels = np.hstack((destrieux["map_left"], destrieux["map_right"]))
fsaverage = datasets.fetch_surf_fsaverage()
surfaces_pial = [fsaverage["pial_left"], fsaverage["pial_right"]]

if run_analysis:
    expression = surface_genetic_expression(labels, surfaces_pial, space="fsaverage")

########################################################################
# Expression is a pandas DataFrame which shows the genetic expression of genes
# within each region of the atlas. By default, the values will fall in the range
# [0, 1] where higher values represent higher expression. However, if you change
# the normalization function then this may change. Some regions may return NaN
# values for all modules. This occurs when there are no samples within this region
# across all donors.
#
# By default, BrainStat uses all the default abagen parameters. If you wish to
# customize these parameters then the keyword arguments can be passed directly to
# `surface_genetic_expression`. For a full list of these arguments and their
# function please consult the abagen documentation.
#
# Meta-Analytic
# -------------
# To perform meta-analytic decoding, BrainStat interfaces with NiMare. Here we
# test which terms are most associated with a map of cortical thickness. A simple example
# analysis can be run as follows. First, we will load some cortical thickness data and
# the white matter surface (recall that we've already loaded the pial surface).

import os
import brainstat
import nibabel as nib
from brainstat.context.meta_analysis import surface_decode_nimare
from brainstat.tutorial.utils import fetch_tutorial_data

## Load white matter surfaces.
surfaces_white = [fsaverage["white_left"], fsaverage["white_right"]]

## Load cortical thickness data.
# Note: you can change the data_dir to wherever you'd like to save the data.
brainstat_dir = os.path.dirname(brainstat.__file__)
data_dir = os.path.join(brainstat_dir, "tutorial")

n = 20
tutorial_data = fetch_tutorial_data(n_subjects=n, data_dir=data_dir)

# Reshape the thickness files such that left and right hemispheres are in the same row.
files = np.reshape(np.array(tutorial_data["image_files"]), (-1, 2))

# We'll use only the left hemisphere in this tutorial.
subject_thickness = np.zeros((n, 20484))
for i in range(n):
    left_thickness = np.squeeze(nib.load(files[i, 0]).get_fdata())
    right_thickness = np.squeeze(nib.load(files[i, 1]).get_fdata())
    subject_thickness[i, :] = np.concatenate((left_thickness, right_thickness))

thickness = np.mean(subject_thickness, axis=0)
mask = np.all(subject_thickness != 0, axis=0)

########################################################################
# Next we can run the analysis. Note that the data and mask has to be
# provided seperately for each hemisphere.

if run_analysis:
    meta_analysis = surface_decode_nimare(
        surfaces_pial,
        surfaces_white,
        [thickness[:10242], thickness[10242:]],
        [mask[:10242], mask[10242:]],
        features=["Neurosynth_TFIDF__visuospatial", "Neurosynth_TFIDF__motor"],
    )

########################################################################
# meta_analysis now contains a pandas.dataFrame with the correlation values
# for each requested feature.
#
# Histological decoding
# ---------------------
# For histological decoding we use microstructural profile covariance gradients
# computed from the BigBrain dataset. (TODO: Add more background). Firstly, lets
# download the MPC data and compute its gradients. As the computations for this aren't
# very intesnive, we can actually run this on ReadTheDocs!

from brainstat.context.histology import (
    read_histology_profile,
    compute_mpc,
    compute_histology_gradients,
)
from brainspace.datasets import load_parcellation

# Load the Schaefer 400 atlas
schaefer_400 = load_parcellation("schaefer", scale=400, join=True)

# Run the analysis
histology_profiles = read_histology_profile(template="fs_LR_64k")
mpc = compute_mpc(histology_profiles, labels=schaefer_400)
gradient_map = compute_histology_gradients(mpc)

########################################################################
# Lets plot the first gradient of histology to see what it looks like.
# We will use BrainSpace to create our plots. For full details on how
# BrainSpace's plotting functionality works, please consult the BrainSpace
# ReadTheDocs. (NOTE: Temporarily disabled due to build errors)

from brainspace.plotting.surface_plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_conte69

left_surface, right_surface = load_conte69()
vertexwise_data = []
for i in range(0, 2):
    vertexwise_data.append(
        map_to_labels(
            gradient_map.gradients_[:, i],
            schaefer_400,
            mask=schaefer_400 != 0,
            fill=np.nan,
        )
    )
# plot_hemispheres(left_surface, right_surface, vertexwise_data, embed_nb=True)
