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
analysis can take several minutes.
"""

import numpy as np
from nilearn import datasets as nil_datasets

from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_template_surface

destrieux = nil_datasets.fetch_atlas_surf_destrieux()
labels = np.hstack((destrieux["map_left"], destrieux["map_right"]))
surfaces = fetch_template_surface("fsaverage5", join=False)

expression = surface_genetic_expression(labels, surfaces, space="fsaverage")
print(expression)

########################################################################
# Expression is a pandas DataFrame which shows the genetic expression of genes
# within each region of the atlas. By default, the values will fall in the range
# [0, 1] where higher values represent higher expression. However, if you change
# the normalization function then this may change. Some regions may return NaN
# values for all modules. This occurs when there are no samples within this
# region across all donors.
#
# By default, BrainStat uses all the default abagen parameters. If you wish to
# customize these parameters then the keyword arguments can be passed directly
# to `surface_genetic_expression`. For a full list of these arguments and their
# function please consult the abagen documentation.
#
# Meta-Analytic
# -------------
# To perform meta-analytic decoding, BrainStat uses precomputed Neurosynth maps.
# Here we test which terms are most associated with a map of cortical thickness.
# A simple example analysis can be run as follows. First, we will load some
# cortical thickness data and two cortical surfaces. The ABIDE dataset provides
# this data on the CIVET surface, so we will also load those surfaces. The
# surface decoder interpolates the data from the surface to the voxels in the
# volume that are in between the two input surfaces.


from brainstat.context.meta_analysis import surface_decoder
from brainstat.datasets import fetch_mask
from brainstat.tutorial.utils import fetch_abide_data

civet_mask = fetch_mask("civet41k")
civet_surface_mid = fetch_template_surface("civet41k", layer="mid", join=False)
civet_surface_white = fetch_template_surface("civet41k", layer="white", join=False)
subject_thickness, demographics = fetch_abide_data(sites=["PITT"])
thickness = subject_thickness.mean(axis=0)

########################################################################
# Next we can run the analysis. Note that the data, surfaces, and mask have to
# be provided seperately for each hemisphere. Also note that downloading the
# dataset and running this analysis can take several minutes.

meta_analysis = surface_decoder(
    civet_surface_mid,
    civet_surface_white,
    [thickness[: len(thickness) // 2], thickness[len(thickness) // 2 :]],
)
print(meta_analysis)

########################################################################
# meta_analysis now contains a pandas.dataFrame with the correlation values
# for each requested feature. If no feature was requested (like here) then
# the analysis is run across all features.
#
#
# Histological decoding
# ---------------------
# For histological decoding we use microstructural profile covariance gradients
# computed from the BigBrain dataset. (TODO: Add more background). Firstly, lets
# download the MPC data and compute its gradients.

from brainstat.context.histology import (
    compute_histology_gradients,
    compute_mpc,
    read_histology_profile,
)
from brainstat.datasets import fetch_parcellation

# Load the Schaefer 400 atlas
schaefer_400 = fetch_parcellation("fsaverage5", "schaefer", 400)

# Run the analysis
histology_profiles = read_histology_profile(template="fsaverage5")
mpc = compute_mpc(histology_profiles, labels=schaefer_400)
gradient_map = compute_histology_gradients(mpc)

########################################################################
# Lets plot the first gradient of histology to see what it looks like.
# We will use BrainSpace to create our plots. For full details on how
# BrainSpace's plotting functionality works, please consult the BrainSpace
# ReadTheDocs.

from brainspace.plotting.surface_plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

surfaces = fetch_template_surface("fsaverage5", join=False)

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

plot_hemispheres(
    surfaces[0],
    surfaces[1],
    vertexwise_data,
    embed_nb=True,
    label_text=["Gradient 1", "Gradient 2"],
)
