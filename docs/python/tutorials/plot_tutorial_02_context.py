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
example of how we use abagen to get the genetic expression of the 400 regions
of the Schaefer atlas. Please note that downloading the dataset and running this
analysis can take several minutes.
"""

import numpy as np

from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_parcellation, fetch_template_surface

schaefer_400 = fetch_parcellation("fsaverage5", "schaefer", 400)
surfaces = fetch_template_surface("fsaverage5", join=False)

expression = surface_genetic_expression(schaefer_400, surfaces, space="fsaverage")
print(expression)

########################################################################
# Expression is a pandas DataFrame which shows the genetic expression of genes
# within each region of the atlas. By default, the values will fall in the range
# [0, 1] where higher values represent higher expression. However, if you change
# the normalization function then this may change. Some regions may return NaN
# values for all genes. This occurs when there are no samples within this
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
# For histological decoding we use microstructural profile covariance gradients,
# as first shown by (Paquola et al, 2019, Plos Biology), computed from the
# BigBrain dataset. Firstly, lets download the MPC data and compute its
# gradients.

from brainstat.context.histology import (
    compute_histology_gradients,
    compute_mpc,
    read_histology_profile,
)
from brainstat.datasets import fetch_parcellation

# Run the analysis
histology_profiles = read_histology_profile(template="fsaverage5")
mpc = compute_mpc(histology_profiles, labels=schaefer_400)
gradient_map = compute_histology_gradients(mpc)

########################################################################
# The variable histology_profiles now contains histological profiles sampled at
# 50 different depths across the cortex, mpc contains the covariance of these
# profiles, and gradient_map contains their gradients. Depending on your
# use-case, each of these variables could be of interest, but for purposes of
# this tutorial we'll plot the gradients to the surface with BrainSpace. For
# details on what the GradientMaps class, gm, contains please consult the
# BrainSpace documentation.

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

########################################################################
# Note that we no longer use the y-axis regression used in (Paquola et al, 2019,
# Plos Biology), as such the first gradient becomes an anterior-posterior-
# gradient.
#
# Resting-state contextualization
# -------------------------------
# Lastly, BrainStat provides contextualization using resting-state fMRI markers:
# specifically, with the Yeo functional networks (Yeo et al., 2011, Journal of
# Neurophysiology), a clustering of resting-state connectivity, and the
# functional gradients (Margulies et al., 2016, PNAS), a lower dimensional
# manifold of resting-state connectivity.
#
# Lets first have a look at contextualization of cortical thickness using the
# Yeo networks. We'll use some of the sample cortical thickness data included
# with BrainSpace, and see what its mean is within each Yeo network. We'll use
# the package plotly to visualize this. plotly is not a dependency of BrainStat
# so you'll have to install it separately (pip install plotly) if you want to
# use this functionality.

import pandas as pd
import plotly.graph_objects as go

from brainspace.datasets import load_marker
from brainspace.utils.parcellation import reduce_by_labels
from brainstat.datasets import fetch_yeo_networks_metadata

thickness = load_marker("thickness", join=True)

yeo_7_networks = fetch_parcellation("fslr32k", "yeo", 7)
network_names, colormap = fetch_yeo_networks_metadata(7)

mean_thickness = reduce_by_labels(thickness, yeo_7_networks, red_op=np.nanmean)[
    1:
]  # 0 == midline

df = pd.DataFrame(mean_thickness[None, :], columns=network_names)
fig = go.Figure(
    data=go.Scatterpolar(r=mean_thickness, theta=network_names, fill="toself")
)

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True),
    ),
    showlegend=False,
)

fig.show(renderer="png")

###########################################################################
# Here we can see that, on average, the somatomotor/visual cortices have low
# cortical thickness whereas the default/limbic cortices have high thickness.
#
# Next, lets have a look at how cortical thickness relates to the first
# functional gradient which describes a sensory-transmodal axis in the brain.
# First lets plot the first few gradients.

from brainstat.datasets import fetch_gradients

functional_gradients = fetch_gradients("fslr32k", "margulies2016")
surface_left, surface_right = fetch_template_surface("fslr32k", join=False)

plot_hemispheres(
    surface_left,
    surface_right,
    functional_gradients[:, 0:3].T,
    embed_nb=True,
    label_text=["Gradient 1", "Gradient 2", "Gradient 3"],
)

###########################################################################
# There are many ways to compare these gradients to cortical markers such as
# cortical thickness. In general, we recommend using corrections for spatial
# autocorrelation which are implemented in BrainSpace. We'll show a correction
# with spin test in this tutorial; for other methods and further details please
# consult the BrainSpace tutorials.
#
# In a spin test we compare the empirical correlation between the gradient and
# the cortical marker to a distribution of correlations derived from data
# rotated across the cortical surface. The p-value then depends on the
# percentile of the empirical correlation within the permuted distribution.

from brainspace.datasets import load_conte69
from brainspace.null_models.spin import spin_permutations

sphere_left, sphere_right = load_conte69(as_sphere=True)
# Note: we generally recommend running >=1000 permutations. For purposes
# of this tutorial we will only run 100.
thickness_permuted_left, thickness_permuted_right = spin_permutations(
    (sphere_left, sphere_right),
    data=np.reshape(thickness, (-1, 2), order="F"),
    n_rep=100,
    random_state=2021,
)

thickness_permuted = np.concatenate(
    (thickness_permuted_left, thickness_permuted_right), axis=1
)

r_empirical = np.corrcoef(functional_gradients[:, 0], thickness)[0, 1]
r_permuted = np.corrcoef(functional_gradients[:, 0], thickness_permuted)[1:, 0]


# Significance depends on whether we do a one-tailed or two-tailed test.
# If one-tailed it depends on in which direction the test is.
p_value_right_tailed = np.mean(r_empirical > r_permuted)
p_value_left_tailed = np.mean(r_empirical < r_permuted)
p_value_two_tailed = np.minimum(p_value_right_tailed, p_value_left_tailed) * 2

###########################################################################
# That concludes the tutorials of BrainStat. If anything is unclear, or if you
# think you've found a bug, please post it to the Issues page of our Github.
#
# Happy BrainStating!
