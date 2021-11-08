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
import plotly.express as px

from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_parcellation, fetch_template_surface

schaefer_400 = fetch_parcellation("fsaverage5", "schaefer", 400)
surfaces = fetch_template_surface("fsaverage5", join=False)

expression = surface_genetic_expression(schaefer_400, surfaces, space="fsaverage")
print(expression.iloc[0:5, 0:5])

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
    color_bar=True,
    size=(1400, 400),
    zoom=1.45,
    nan_color=(0.7, 0.7, 0.7, 1),
    cb__labelTextProperty={"fontSize": 12},
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
# As an example, lets have a look at the first functional gradient within the
# Yeo networks.
#

import matplotlib.pyplot as plt

from brainstat.context.resting import yeo_networks_average
from brainstat.datasets import fetch_gradients, fetch_yeo_networks_metadata

functional_gradients = fetch_gradients("fslr32k", "margulies2016")
yeo_gradients = yeo_networks_average(functional_gradients, "fslr32k")
network_names, yeo_colormap = fetch_yeo_networks_metadata(7)

plt.bar(np.arange(7), yeo_gradients[:, 0], color=yeo_colormap)
plt.xticks(np.arange(7), network_names, rotation=90)
plt.gcf().subplots_adjust(bottom=0.3)
plt.show()

###########################################################################
# Unsurprisingly, the gradients are very similar to the Yeo networks, with
# higher gradient values in higher order networks and lower values in lower
# order networks.
#
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

from brainspace.datasets import load_conte69, load_marker
from brainspace.null_models import SpinPermutations

sphere_left, sphere_right = load_conte69(as_sphere=True)
thickness_left, thickness_right = load_marker("thickness", join=False)
thickness = load_marker("thickness", join=True)

# Run spin test with 100 permutations (note: we generally recommend >=1000)
n_rep = 100
sp = SpinPermutations(n_rep=n_rep, random_state=2021)
sp.fit(sphere_left, points_rh=sphere_right)
thickness_rotated = np.hstack(sp.randomize(thickness_left, thickness_right))

# Compute correlation between empirical and permuted data.
mask = ~np.isnan(functional_gradients[:, 0]) & ~np.isnan(thickness)
r_empirical = np.corrcoef(functional_gradients[mask, 0], thickness[mask])[0, 1]
r_permuted = np.zeros(n_rep)
for i in range(n_rep):
    mask = ~np.isnan(functional_gradients[:, 0]) & ~np.isnan(thickness_rotated[i, :])
    r_permuted[i] = np.corrcoef(
        functional_gradients[mask, 0], thickness_rotated[i, mask]
    )[1:, 0]

# Significance depends on whether we do a one-tailed or two-tailed test.
# If one-tailed it depends on in which direction the test is.
p_value_right_tailed = np.mean(r_empirical > r_permuted)
p_value_left_tailed = np.mean(r_empirical < r_permuted)
p_value_two_tailed = np.minimum(p_value_right_tailed, p_value_left_tailed) * 2
print(f"Two tailed p-value: {p_value_two_tailed}")

###########################################################################
# That concludes the tutorials of BrainStat. If anything is unclear, or if you
# think you've found a bug, please post it to the Issues page of our Github.
#
# Happy BrainStating!
