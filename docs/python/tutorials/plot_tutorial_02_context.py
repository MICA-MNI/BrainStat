"""
Tutorial 02: Context Decoding
=========================================

In this tutorial you will learn about the context decoding tools included with
BrainStat. The context decoding module consists of three parts: genetic
decoding, meta-analytic decoding and histological comparisons. Before we start,
lets run a linear model testing for the effects of age on cortical thickness as
we did in Tutorial 1. We'll use the results of this model later in this
tutorial.
"""

import numpy as np

from brainstat.datasets import fetch_mask, fetch_template_surface
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect
from brainstat.tutorial.utils import fetch_abide_data

sites = ("PITT", "OLIN", "OHSU")
thickness, demographics = fetch_abide_data(sites=sites)
mask = fetch_mask("civet41k")

demographics.DX_GROUP[demographics.DX_GROUP == 1] = "Patient"
demographics.DX_GROUP[demographics.DX_GROUP == 2] = "Control"

term_age = FixedEffect(demographics.AGE_AT_SCAN)
term_patient = FixedEffect(demographics.DX_GROUP)
model = term_age + term_patient

contrast_age = model.AGE_AT_SCAN
slm_age = SLM(
    model, contrast_age, surf="civet41k", mask=mask, correction=["fdr", "rft"]
)
slm_age.fit(thickness)

###################################################################
# Genetics
# --------
#
# For genetic decoding we use the Allen Human Brain Atlas through the abagen
# toolbox. Note that abagen only accepts parcellated data. Here is a minimal
# example of how we use abagen to get the genetic expression of the 400 regions
# of the Schaefer atlas. Please note that downloading the dataset and running this
# analysis can take several minutes.

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_parcellation

# Get Schaefer-100 genetic expression.
schaefer_100 = fetch_parcellation("fsaverage5", "schaefer", 100)
surfaces = fetch_template_surface("fsaverage5", join=False)
expression = surface_genetic_expression(schaefer_100, surfaces, space="fsaverage")

# Plot Schaefer-100 genetic expression matrix.
colormap = copy.copy(get_cmap())
colormap.set_bad(color="black")
plt.imshow(expression, aspect="auto", cmap=colormap)
plt.colorbar()
plt.xlabel("Genetic Expression")
plt.ylabel("Schaefer 100 Regions")
plt.show()

########################################################################
# Expression is a pandas DataFrame which shows the genetic expression of genes
# within each region of the atlas. By default, the values will fall in the range
# [0, 1] where higher values represent higher expression. However, if you change
# the normalization function then this may change. Some regions may return NaN
# values for all genes. This occurs when there are no samples within this
# region across all donors. We've denoted this region with the black color in the
# matrix.
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
# A simple example analysis can be run as follows. The surface decoder
# interpolates the data from the surface to the voxels in the volume that are in
# between the two input surfaces. We'll decode the t-statistics derived with our model
# earlier.


from brainstat.context.meta_analysis import surface_decoder

civet_mask = fetch_mask("civet41k")
civet_surface_mid = fetch_template_surface("civet41k", layer="mid", join=False)
civet_surface_white = fetch_template_surface("civet41k", layer="white", join=False)

########################################################################
# Next we can run the analysis. Note that the data, surfaces, and mask have to
# be provided seperately for each hemisphere. Also note that downloading the
# dataset and running this analysis can take several minutes.
t_stats = np.squeeze(slm_age.t)
meta_analysis = surface_decoder(
    civet_surface_mid,
    civet_surface_white,
    [t_stats[: t_stats.size // 2], t_stats[t_stats.size // 2 :]],
)

print(meta_analysis)

##########################################################################
# meta_analysis now contains a pandas.dataFrame with the correlation values for
# each requested feature. Printing all of the terms will return many terms that
# may not be of interest (e.g. anatomical regions). Lets select a few terms of
# interest. This may be done as follows:

terms_of_interest = [
    "attention",
    "emotion",
    "language comprehension",
    "motor",
    "reading",
    "semantics",
    "social cognition",
    "spatial attention",
    "speech production",
    "visual perception",
    "visuospatial",
]

meta_analysis_subset = meta_analysis.loc[terms_of_interest, :]
print(meta_analysis_subset.sort_values(by="Pearson's r", ascending=False))


########################################################################
# Here we see that language comprehension and spatial attention correlate
# moderately, in opposite directions, with our t-map. Several other correlations
# are also shown.
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

# Run the analysis
schaefer_400 = fetch_parcellation("fsaverage5", "schaefer", 400)
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

# Bring parcellated data to vertex data.
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

# Plot to surface.
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


import matplotlib.pyplot as plt

from brainstat.context.resting import yeo_networks_associations
from brainstat.datasets import fetch_yeo_networks_metadata

yeo_tstat = yeo_networks_associations(np.squeeze(slm_age.t), "civet41k")
network_names, yeo_colormap = fetch_yeo_networks_metadata(7)

plt.bar(np.arange(7), yeo_tstat[:, 0], color=yeo_colormap)
plt.xticks(np.arange(7), network_names, rotation=90)
plt.gcf().subplots_adjust(bottom=0.3)
plt.show()


###########################################################################
# Across all networks, the mean t-statistic appears to be negative, with the
# most negative values in the dorsal attnetion and visual networks.
#
# Lastly, lets plot the functional gradients and have a look at their correlation
# with our t-map.


import pandas as pd

from brainstat.datasets import fetch_gradients

surfaces_civet = fetch_template_surface("civet41k", join=False)
functional_gradients = fetch_gradients("civet41k", "margulies2016")

plot_hemispheres(
    surfaces_civet[0],
    surfaces_civet[1],
    functional_gradients[:, 0:3].T,
    color_bar=True,
    label_text=["Gradient 1", "Gradient 2", "Gradient 3"],
    embed_nb=True,
    size=(1400, 600),
    zoom=1.45,
    cb__labelTextProperty={"fontSize": 12},
)

###########################################################################

r = pd.DataFrame(functional_gradients[:, 0:3]).corrwith(pd.Series(slm_age.t.flatten()))
print(r)


###########################################################################
# It seems the correlations are, overall quite low. However, we'll need some
# more complex tests to assess statistical significance. There are many ways to
# compare these gradients to cortical markerss. In general, we recommend using
# corrections for spatial autocorrelation which are implemented in BrainSpace.
# We'll show a correction with spin test in this tutorial; for other methods and
# further details please consult the BrainSpace tutorials.
#
# In a spin test we compare the empirical correlation between the gradient and
# the cortical marker to a distribution of correlations derived from data
# rotated across the cortical surface. The p-value then depends on the
# percentile of the empirical correlation within the permuted distribution.
# As we do not have a CIVET sphere included with BrainStat, we'll use BrainSpace's
# template data on fslr32k.


from brainspace.datasets import load_conte69, load_marker
from brainspace.null_models import SpinPermutations

sphere_left, sphere_right = load_conte69(as_sphere=True)
thickness_left, thickness_right = load_marker("thickness", join=False)
thickness = load_marker("thickness", join=True)
functional_gradients_fslr = fetch_gradients("fslr32k", "margulies2016")

# Run spin test with 100 permutations (note: we generally recommend >=1000)
n_rep = 100
sp = SpinPermutations(n_rep=n_rep, random_state=2021)
sp.fit(sphere_left, points_rh=sphere_right)
thickness_rotated = np.hstack(sp.randomize(thickness_left, thickness_right))

# Compute correlation between empirical and permuted data.
mask = ~np.isnan(functional_gradients_fslr[:, 0]) & ~np.isnan(thickness)
r_empirical = np.corrcoef(functional_gradients_fslr[mask, 0], thickness[mask])[0, 1]
r_permuted = np.zeros(n_rep)
for i in range(n_rep):
    mask = ~np.isnan(functional_gradients_fslr[:, 0]) & ~np.isnan(
        thickness_rotated[i, :]
    )
    r_permuted[i] = np.corrcoef(
        functional_gradients_fslr[mask, 0], thickness_rotated[i, mask]
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
