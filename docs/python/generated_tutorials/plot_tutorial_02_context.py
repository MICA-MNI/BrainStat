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

from brainstat.datasets import fetch_mask, fetch_template_surface
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import FixedEffect, MixedEffect
from brainstat.tutorial.utils import fetch_mics_data

thickness, demographics = fetch_mics_data()
mask = fetch_mask("fsaverage5")

term_age = FixedEffect(demographics.AGE_AT_SCAN)
term_sex = FixedEffect(demographics.SEX)
term_subject = MixedEffect(demographics.SUB_ID)
model = term_age + term_sex + term_age * term_sex + term_subject

contrast_age = -model.mean.AGE_AT_SCAN
slm = SLM(
    model,
    contrast_age,
    surf="fsaverage5",
    mask=mask,
    correction=["fdr", "rft"],
    two_tailed=False,
    cluster_threshold=0.01,
)
slm.fit(thickness)

###################################################################
# Genetics
# --------
#
# For genetic decoding we use the Allen Human Brain Atlas through the abagen
# toolbox. Note that abagen only accepts parcellated data. Here is a minimal
# example of how we use abagen to get the genetic expression of the 100 regions
# of the Schaefer atlas and how to plot this expression to a matrix. Please note
# that downloading the dataset and running this analysis can take several
# minutes.

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.utils.parcellation import reduce_by_labels
from matplotlib.cm import get_cmap

from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_parcellation

# Get Schaefer-100 genetic expression.
schaefer_100_fs5 = fetch_parcellation("fsaverage5", "schaefer", 100)
surfaces = fetch_template_surface("fsaverage5", join=False)
expression = surface_genetic_expression(schaefer_100_fs5, surfaces, space="fsaverage")

# Plot Schaefer-100 genetic expression matrix.
colormap = copy.copy(get_cmap())
colormap.set_bad(color="black")
plt.imshow(expression, aspect="auto", cmap=colormap)
plt.colorbar().ax.tick_params(labelsize=14)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.xlabel("Gene Index", fontdict={"fontsize": 16})
plt.ylabel("Schaefer 100 Regions", fontdict={"fontsize": 16})
plt.gcf().subplots_adjust(bottom=0.2)

########################################################################
# Expression is a pandas DataFrame which shows the genetic expression of genes
# within each region of the atlas. By default, the values will fall in the range
# [0, 1] where higher values represent higher expression. However, if you change
# the normalization function then this may change. Some regions may return NaN
# values for all genes. This occurs when there are no samples within this
# region across all donors. We've denoted this region with the black color in the
# matrix. By default, BrainStat uses all the default abagen parameters. If you wish to
# customize these parameters then the keyword arguments can be passed directly
# to `surface_genetic_expression`. For a full list of these arguments and their
# function please consult the abagen documentation.
#
# Next, lets have a look at the correlation between one gene (WFDC1) and our
# t-statistic map. Lets also plot the expression of this gene to the surface.

# Plot correlation with WFDC1 gene
t_stat_schaefer_100 = reduce_by_labels(slm.t.flatten(), schaefer_100_fs5)[1:]

df = pd.DataFrame({"x": t_stat_schaefer_100, "y": expression["WFDC1"]})
df.dropna(inplace=True)
plt.scatter(df.x, df.y, s=20, c="k")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("t-statistic", fontdict={"fontsize": 16})
plt.ylabel("WFDC1 expression", fontdict={"fontsize": 16})
plt.plot(np.unique(df.x), np.poly1d(np.polyfit(df.x, df.y, 1))(np.unique(df.x)), "k")
plt.text(-1.0, 0.75, f"r={df.x.corr(df.y):.2f}", fontdict={"size": 14})
plt.show()

########################################################################

# Plot WFDC1 gene to the surface.
from brainspace.plotting.surface_plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

vertexwise_WFDC1 = map_to_labels(
    expression["WFDC1"].to_numpy(),
    schaefer_100_fs5,
    mask=schaefer_100_fs5 != 0,
    fill=np.nan,
)

plot_hemispheres(
    surfaces[0],
    surfaces[1],
    vertexwise_WFDC1,
    color_bar=True,
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    nan_color=(0.7, 0.7, 0.7, 1),
    cb__labelTextProperty={"fontSize": 12},
)

########################################################################
# We find a small correlation. To test for significance we'll have
# to do some additional corrections, but more on that later.
#
# Meta-Analytic
# -------------
# To perform meta-analytic decoding, BrainStat uses precomputed Neurosynth maps.
# Here we test which terms are most associated with a map of cortical thickness.
# A simple example analysis can be run as follows. The surface decoder
# interpolates the data from the surface to the voxels in the volume that are in
# between the two input surfaces. We'll decode the t-statistics derived with our model
# earlier. Note that downloading the dataset and running this analysis can take several minutes.

from brainstat.context.meta_analysis import meta_analytic_decoder

meta_analysis = meta_analytic_decoder("fsaverage5", slm.t.flatten())
print(meta_analysis)

##########################################################################
# meta_analysis now contains a pandas.dataFrame with the correlation values for
# each requested feature. Next we could create a Wordcloud of the included terms,
# wherein larger words denote higher correlations.
from wordcloud import WordCloud

wc = WordCloud(background_color="white", random_state=0)
wc.generate_from_frequencies(frequencies=meta_analysis.to_dict()["Pearson's r"])
plt.imshow(wc)
plt.axis("off")
plt.show()

########################################################################
# Alternatively, we can visualize the top correlation values and associated terms
# in a radar plot, as follows:
from brainstat.context.meta_analysis import radar_plot

numFeat = 8
data = meta_analysis.to_numpy()[:numFeat]
label = meta_analysis.index[:numFeat]
radar_plot(data, label=label, axis_range=(0.18, 0.22))

########################################################################
# If we broadly summarize, we see a lot of words related to language e.g.,
# "language comprehension", "broca", "speaking", "speech production".
# Generally you'll also find several hits related to anatomy or clinical conditions.
# Depending on your research question, it may be more interesting to
# select only those terms related to cognition or some other subset.

########################################################################
# Histological decoding
# ---------------------
# For histological decoding we use microstructural profile covariance gradients,
# as first shown by (Paquola et al, 2019, Plos Biology), computed from the
# BigBrain dataset. Firstly, lets download the MPC data, compute and plot its
# gradients, and correlate the first gradient with our t-statistic map.

from brainstat.context.histology import (
    compute_histology_gradients,
    compute_mpc,
    read_histology_profile,
)

# Run the analysis
schaefer_400 = fetch_parcellation("fsaverage5", "schaefer", 400)
histology_profiles = read_histology_profile(template="fsaverage5")
mpc = compute_mpc(histology_profiles, labels=schaefer_400)
gradient_map = compute_histology_gradients(mpc, random_state=0)

# Bring parcellated data to vertex data.
vertexwise_gradient = map_to_labels(
    gradient_map.gradients_[:, 0],
    schaefer_400,
    mask=schaefer_400 != 0,
    fill=np.nan,
)

plot_hemispheres(
    surfaces[0],
    surfaces[1],
    vertexwise_gradient,
    embed_nb=True,
    nan_color=(0.7, 0.7, 0.7, 1),
    size=(1400, 200),
    zoom=1.45,
)

########################################################################

# Plot the correlation between the t-stat
t_stat_schaefer_400 = reduce_by_labels(slm.t.flatten(), schaefer_400)[1:]
df = pd.DataFrame({"x": t_stat_schaefer_400, "y": gradient_map.gradients_[:, 0]})
df.dropna(inplace=True)
plt.scatter(df.x, df.y, s=5, c="k")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("t-statistic", fontdict={"fontsize": 16})
plt.ylabel("MPC Gradient 1", fontdict={"fontsize": 16})
plt.plot(np.unique(df.x), np.poly1d(np.polyfit(df.x, df.y, 1))(np.unique(df.x)), "k")
plt.text(2.3, 0.1, f"r={df.x.corr(df.y):.2f}", fontdict={"size": 14})
plt.gcf().subplots_adjust(left=0.15)
plt.show()

########################################################################
# The variable histology_profiles now contains histological profiles sampled at
# 50 different depths across the cortex, mpc contains the covariance of these
# profiles, and gradient_map contains their gradients. We also see that the
# correlation between our t-statistic map and these gradients is not very
# high. Depending on your use-case, each of the three variables here could be of
# interest, but for purposes of this tutorial we'll plot the gradients to the
# surface with BrainSpace. For details on what the GradientMaps class
# (gradient_map) contains please consult the BrainSpace documentation.

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
# Plos Biology), as such the first gradient becomes an anterior-posterior
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
# As an example, lets have a look at the the t-statistic map within the Yeo
# networks. We'll plot the Yeo networks as well as a barplot showing the mean
# and standard error of the mean within each network.
from brainstat.datasets import fetch_yeo_networks_metadata

yeo_networks = fetch_parcellation("fsaverage5", "yeo", 7)
network_names, yeo_colormap = fetch_yeo_networks_metadata(7)

plot_hemispheres(
    surfaces[0],
    surfaces[1],
    yeo_networks,
    embed_nb=True,
    cmap="yeo7",
    nan_color=(0.7, 0.7, 0.7, 1),
    size=(1400, 200),
    zoom=1.45,
)

##########################################################################
import matplotlib.pyplot as plt
from scipy.stats import sem

from brainstat.context.resting import yeo_networks_associations

yeo_tstat_mean = yeo_networks_associations(slm.t.flatten(), "fsaverage5")
yeo_tstat_sem = yeo_networks_associations(
    slm.t.flatten(),
    "fsaverage5",
    reduction_operation=lambda x, y: sem(x, nan_policy="omit"),
)

plt.bar(
    np.arange(7),
    yeo_tstat_mean[:, 0],
    yerr=yeo_tstat_sem.flatten(),
    color=yeo_colormap,
    error_kw={"elinewidth": 5},
)
plt.xticks(np.arange(7), network_names, rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("t-statistic", fontdict={"fontsize": 16})
plt.gcf().subplots_adjust(left=0.2, bottom=0.5)
plt.show()

###########################################################################
# Across all networks, the mean t-statistic appears to be negative, with the
# most negative values in the dorsal attnetion and visual networks.
#
# Lastly, lets plot the functional gradients and have a look at their correlation
# with our t-map.

from brainstat.datasets import fetch_gradients

functional_gradients = fetch_gradients("fsaverage5", "margulies2016")


plot_hemispheres(
    surfaces[0],
    surfaces[1],
    functional_gradients[:, 0:3].T,
    color_bar=True,
    label_text=["Gradient 1", "Gradient 2", "Gradient 3"],
    embed_nb=True,
    size=(1400, 600),
    zoom=1.45,
    nan_color=(0.7, 0.7, 0.7, 1),
    cb__labelTextProperty={"fontSize": 12},
)

###########################################################################

df = pd.DataFrame({"x": slm.t.flatten(), "y": functional_gradients[:, 0]})
df.dropna(inplace=True)
plt.scatter(df.x, df.y, s=0.01, c="k")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("t-statistic", fontdict={"fontsize": 16})
plt.ylabel("Functional Gradient 1", fontdict={"fontsize": 16})
plt.plot(np.unique(df.x), np.poly1d(np.polyfit(df.x, df.y, 1))(np.unique(df.x)), "k")
plt.text(-4.0, 6, f"r={df.x.corr(df.y):.2f}", fontdict={"size": 14})
plt.gcf().subplots_adjust(left=0.2)
plt.show()


###########################################################################
# It seems the correlations are quite low. However, we'll need some more complex
# tests to assess statistical significance. There are many ways to compare these
# gradients to cortical markers. In general, we recommend using corrections for
# spatial autocorrelation which are implemented in BrainSpace. We'll show a
# correction with spin test in this tutorial; for other methods and further
# details please consult the BrainSpace tutorials.
#
# In a spin test we compare the empirical correlation between the gradient and
# the cortical marker to a distribution of correlations derived from data
# rotated across the cortical surface. The p-value then depends on the
# percentile of the empirical correlation within the permuted distribution.


from brainspace.null_models import SpinPermutations

sphere_left, sphere_right = fetch_template_surface(
    "fsaverage5", layer="sphere", join=False
)
tstat = slm.t.flatten()
tstat_left = tstat[: slm.t.size // 2]
tstat_right = tstat[slm.t.size // 2 :]

# Run spin test with 1000 permutations.
n_rep = 1000
sp = SpinPermutations(n_rep=n_rep, random_state=2021)
sp.fit(sphere_left, points_rh=sphere_right)
tstat_rotated = np.hstack(sp.randomize(tstat_left, tstat_right))

# Compute correlation for empirical and permuted data.
mask = ~np.isnan(functional_gradients[:, 0]) & ~np.isnan(tstat)
r_empirical = np.corrcoef(functional_gradients[mask, 0], tstat[mask])[0, 1]
r_permuted = np.zeros(n_rep)
for i in range(n_rep):
    mask = ~np.isnan(functional_gradients[:, 0]) & ~np.isnan(tstat_rotated[i, :])
    r_permuted[i] = np.corrcoef(functional_gradients[mask, 0], tstat_rotated[i, mask])[
        1:, 0
    ]

# Significance depends on whether we do a one-tailed or two-tailed test.
# If one-tailed it depends on in which direction the test is.
p_value_right_tailed = np.mean(r_empirical > r_permuted)
p_value_left_tailed = np.mean(r_empirical < r_permuted)
p_value_two_tailed = np.minimum(p_value_right_tailed, p_value_left_tailed) * 2
print(f"Two tailed p-value: {p_value_two_tailed}")

# Plot the permuted distribution of correlations.
plt.hist(r_permuted, bins=20, color="c", edgecolor="k", alpha=0.65)
plt.axvline(r_empirical, color="k", linestyle="dashed", linewidth=1)
plt.show()

###########################################################################
# As we can see from both the p-value as well as the histogram, wherein the
# dotted line denotes the empirical correlation, this correlation does not reach
# significance.
#
# That concludes the tutorials of BrainStat. If anything is unclear, or if you
# think you've found a bug, please post it to the Issues page of our Github.
#
# Happy BrainStating!
