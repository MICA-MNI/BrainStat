"""
Tutorial 01: Linear Models
=========================================
In this tutorial you will set up your first linear model with BrainStat. 
To this end, we will load some sample data from the ABIDE dataset. Note that,
contrary to the results shown in our manuscript, we are only using a few sites
to reduce computation time in this tutorial. As such the results shown here
differ from those reported in our manuscript.
"""


import numpy as np
from brainstat.datasets import fetch_template_surface
from brainstat.tutorial.utils import fetch_abide_data

# Load behavioral markers
sites = ("PITT", "OLIN", "OHSU")
thickness, demographics = fetch_abide_data(sites=sites)
pial_left, pial_right = fetch_template_surface("civet41k", join=False)
pial_combined = fetch_template_surface("civet41k", join=True)

###################################################################
# Lets have a look at the cortical thickness data. To do this,
# we will use the surface plotter included with BrainSpace. Lets plot
# mean thickness. 
from brainspace.plotting import plot_hemispheres

plot_hemispheres(
    pial_left,
    pial_right,
    np.mean(thickness, axis=0),
    color_bar=True,
    color_range=(1.5, 3.5),
    label_text=["Cortical Thickness"],
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)

###################################################################
# Next, lets see whether cortical thickness is related to age in our sample
# data. To this end we can create a linear model with BrainStat. First we
# declare the behavioral variables as FixedEffects. The FixedEffect class can be
# created in two ways: either we provide the data with pandas, as we do here, or
# we provide a numpy array and a name for the fixed effect. Once, that's done we
# can create the model by simply adding the terms together. Lets set up a model
# with age and patient status as fixed effects.

from brainstat.stats.terms import FixedEffect

term_age = FixedEffect(demographics.AGE_AT_SCAN)
# Subtract 1 from DX_GROUP so patient == 0 and healthy == 1.
term_patient = FixedEffect(demographics.DX_GROUP - 1) 
model = term_age + term_patient

###################################################################
# As said before, if your data is not in a pandas DataFrame (e.g. numpy), you'll
# have to provide the name of the effect as an additional parameter as follows:
term_age_2 = FixedEffect(demographics.AGE_AT_SCAN.to_numpy(), "AGE_AT_SCAN")

###################################################################
# Beside simple fixed effects, we may also be interested in interaction
# effects. We can add these to the model by multiplying terms. Lets
# create a model with an interaction between age and patient status.

model_interaction = term_age + term_patient + term_age * term_patient

###################################################################
# Lets have a look at one of these models. As you can see below, the model
# is stored in a format closely resembling a pandas DataFrame. Note that an
# intercept is automatically added to the model. This behavior can be disabled
# in the FixedEffect call, but we recommend leaving it enabled.

print(model)

###################################################################
# Now, imagine we have some cortical marker (e.g. cortical thickness) for
# each subject, and we want to evaluate whether this marker changes with age
# whilst correcting for effects of patient status. To do this, we can use
# the model we defined before, and a contrast in observations (here: age).
# Then we simply initialize an SLM model and fit it to the cortical thickness
# data.

from brainstat.stats.SLM import SLM

contrast_age = model.AGE_AT_SCAN
slm_age = SLM(model, contrast_age, surf=pial_combined, correction="rft")
slm_age.fit(thickness)

plot_hemispheres(
    pial_left,
    pial_right,
    slm_age.t,
    label_text=["t-values"],
    color_bar=True,
    color_range=(-4, 4),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)
plot_hemispheres(
    pial_left,
    pial_right,
    slm_age.P["pval"]["P"],
    label_text=["p-values"],
    color_bar=True,
    color_range=(0, 0.05),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)


###################################################################
# By default BrainStat uses a two-tailed test. If you want to get a one-tailed
# test, simply specify it in the SLM model initialization with
# `two_tailed=True`. Note that the one-tailed test will test for positive
# t-values. If you want to test for negative t-values, simply invert the
# contrast. We may hypothesize based on prior research that cortical thickness
# decreases with age, so we could specify this as follows:

# Note the minus in front of contrast_age to test for decreasing thickness with age.
slm_age_onetailed = SLM(
    model, -contrast_age, surf=pial_combined, correction="rft", two_tailed=False
)
slm_age_onetailed.fit(thickness)

plot_hemispheres(
    pial_left,
    pial_right,
    slm_age_onetailed.t,
    label_text=["t-values"],
    color_bar=True,
    color_range=(-4, 4),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)
plot_hemispheres(
    pial_left,
    pial_right,
    slm_age_onetailed.P["pval"]["P"],
    label_text=["p-values"],
    color_bar=True,
    color_range=(0, 0.05),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)


###################################################################
# Similarly, we could perform an analysis to assess cortical thickness
# differences across healthy and patient groups whilst correcting for age.

contrast_patient = model.DX_GROUP
slm_patient = SLM(model, contrast_patient, surf=pial_combined, correction="rft")
slm_patient.fit(thickness)

plot_hemispheres(
    pial_left,
    pial_right,
    slm_patient.t,
    label_text=["t-values"],
    color_bar=True,
    color_range=(-4, 4),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)
plot_hemispheres(
    pial_left,
    pial_right,
    slm_patient.P["pval"]["P"],
    label_text=["p-values"],
    color_bar=True,
    color_range=(0, 0.05),
    embed_nb=True,
    size=(1400, 200),
    zoom=1.45,
    cb__labelTextProperty = {'fontSize': 12},
)


###################################################################
# It appears we do not find statistically significant results for a relationship
# between cortical thickness and patient status.
#
# Now, imagine that instead of using a fixed effects model, you would prefer a
# mixed effects model wherein the scanning site is a random variable. This is
# simple to set up. All you need to do is initialize the site term with the
# MixedEffect class, all other code remains identical. 

from brainstat.stats.terms import MixedEffect

random_site = MixedEffect(demographics.SITE_ID, name_ran="Site")

model_random = term_age + term_patient + random_site
slm_random = SLM(model_random, contrast_age, surf=pial_left, correction="rft")
slm_random.fit(thickness)

###############################################################################
# Lets have a closer look at the mixed effect. The variable random_site contains
# two important properties: "mean", and "variance". "mean" contains any fixed effects,
# whereas "variance" contains the random effects. As we did not specify any fixed
# effects, the "mean" term is empty. The "variance" term contains the random effect as
# well as the identity term, similar to the intercept in FixedEffects. The identity term
# is added by default.

print(random_site.mean)
print(random_site.variance)
