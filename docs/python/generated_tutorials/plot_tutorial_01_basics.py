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
# Lets have a look at the data that we have loaded. For this, we'll use the
# surface plotter included with BrainSpace.
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
# So, next, lets see whether cortical thickness is related to age in our sample
# data. To this end we can create a BrainStat linear model. First we declare the
# behavioral variables as FixedEffects. The FixedEffect class requires two
# things: 1) an array or scalar, and 2) a variable name for each column. Once,
# that's done we can create the model by simply adding the terms together.
# Lets set up a model with age and IQ as fixed effects.

from brainstat.stats.terms import FixedEffect

term_age = FixedEffect(demographics.AGE_AT_SCAN)
term_patient = FixedEffect(
    demographics.DX_GROUP - 1
)  # Subtract 1 so patient==0, control==1
model = term_age + term_patient

# Note: if your data is not in a pandas DataFrame (e.g. numpy), you'll have
# to provide the name of the effect as an additional parameter as follows:
term_age_2 = FixedEffect(demographics.AGE_AT_SCAN.to_numpy(), "AGE_AT_SCAN")

###################################################################
# We can also add interaction effects to the model by multiplying terms. Lets
# add an interaction between age and sex.

model_interaction = term_age + term_patient + term_age * term_patient

###################################################################
# Lets have a look at one of these models. As you can see below, the model
# is stored in a format closely resembling a pandas DataFrame. Note that an
# intercept is automatically added to the model. This behavior can be disabled
# in the FixedEffect call, but we recommend leaving it enabled.

print(model)

# The interaction model also contains the interaction term:

print(model_interaction)

###################################################################
# Now, imagine we have some cortical marker (e.g. cortical thickness) for
# each subject, and we want to evaluate whether this marker changes with age
# whilst correcting for effects of healthy / patient status.

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
# mixed effects model wherein site is a random variable. This is simple to
# set up. All you need to do is initialize the site term with the MixedEffect
# class instead, all other code remains identical. As site is a categorical
# variable, we'll have to transform it into a dummy variable first.

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
