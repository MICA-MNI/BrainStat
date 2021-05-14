"""
Tutorial 02: Context Decoding
=========================================

In this tutorial you will learn about the context decoding tools included with
BrainStat. The context decoding module consists of three parts: genetic
deocding, meta-analytic decoding and histological comparisons. First, we'll
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
# test which terms occur more often in manuscripts that report activation with the
# target region, as opposed to regions reported in any part of the cortex. Please
# be aware that this analysis currently uses a lot of memory (~80GB). A simple example
# analysis can be run as follows:

from brainstat.context.meta_analysis import surface_decode_neurosynth

surfaces_white = [fsaverage["white_left"], fsaverage["white_right"]]
roi = [labels[0:10242] == 1, labels[10242:] == 1]
all_cortex = [labels[0:10242] > 0, labels[10242:]]

if run_analysis:
    meta_analysis = surface_decode_neurosynth(
        surfaces_pial, surfaces_white, roi, all_cortex
    )
