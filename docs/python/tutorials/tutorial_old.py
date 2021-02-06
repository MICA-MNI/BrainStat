"""
Tutorial 01: Linear Models
=========================================

In this tutorial you will set up your first linear model with SurfStat. Please
note that brainstat does not include sample data yet. Once we introduce
example data loader functions into BrainStat this tutorial will be modified
accordingly. Until such time, we will rely on randomly generated data and sample
data from BrainSpace.

Setting up a linear model
---------------------------------
Lets start setting up a basic model. Recall that fixed linear models take the
form of :math:`Y = \\beta_0 + \\beta_1x_1 + ... + \\beta_nx_n + \\varepsilon` where
:math:`\\beta_0` denotes the intercept, :math:`\\beta_i` denotes the slope for
variable :math:`x_i`, and :math:`\\varepsilon` is the error term. In BrainStat we
can easily set up such a model as follows.

First lets load some example data to play around with. We'll load age, IQ, and left
hemispheric cortical thickness for a few subjects.
"""


###################################################################

import numpy as np
import nibabel as nib
import os
import brainstat
from brainstat.tutorial.utils import fetch_tutorial_data
from brainstat.context.utils import read_surface_gz
from nilearn.datasets import fetch_surf_fsaverage

brainstat_dir = os.path.dirname(brainstat.__file__)
data_dir = os.path.join(brainstat_dir, "tutorial")

n = 20
tutorial_data = fetch_tutorial_data(n_subjects=n, data_dir=data_dir)
age = tutorial_data["demographics"]["AGE"].to_numpy()
iq = tutorial_data["demographics"]["IQ"].to_numpy()

# Reshape the thickness files such that left and right hemispheres are in the same row.
files = np.reshape(np.array(tutorial_data["image_files"]), (-1, 2))

# We'll use only the left hemisphere in this tutorial.
thickness = np.zeros((n, 10242))
for i in range(n):
    thickness[i, :] = np.squeeze(nib.load(files[i, 0]).get_fdata())

pial_left = read_surface_gz(fetch_surf_fsaverage()["pial_left"])


###################################################################
# Next, we can create a BrainStat linear model by declaring these variables as
# terms. The term class requires two things: 1) an array or scalar, and 2) a
# variable name for each column. Lastly, we can create the model by simply
# adding the terms together.


###################################################################


from brainstat.stats.terms import Term

term_intercept = Term(1, names="intercept")
term_age = Term(age, "age")
term_iq = Term(iq, "iq")
model = term_intercept + term_age + term_iq


###################################################################
# We can also add interaction effects to the model by multiplying terms.


###################################################################


model_interaction = term_intercept + term_age + term_iq + term_age * term_iq


###################################################################
# Now, lets imagine we have some cortical marker (e.g. cortical thickness) for
# each subject and we want to evaluate whether this marker changes with age
# whilst correcting for effects of sex and age-sex interactions. Note that
# BrainStat's univariate tests are one-tailed, so the sign of the contrast
# matters!


###################################################################


from brainstat.stats.models import linear_model, t_test

Y = np.random.rand(n, 10242)  # Surface has 10242 vertices.
slm = linear_model(Y, model_interaction, pial_left)
slm = t_test(slm, -age)
print(slm["t"])  # These are the t-values of the model.


###################################################################
# Never forget: with great models come great multiple comparisons corrections.
# BrainStat provides two methods for these corrections: FDR and random field theory.
# In this example we'll show you how to use random field theory to find significant
# results at alpha=0.05.


###################################################################

from brainstat.stats.multiple_comparisons import random_field_theory

alpha = 0.05
P, _, _, _ = random_field_theory(slm)
print(P["P"] < alpha)

###################################################################
# As said before, univariate tests in BrainStat use a one-tailed test. If you
# want to get a two-tailed text, simply run contrast as well as its negative and
# adjust the alpha accordingly.


###################################################################

slm_basic = linear_model(Y, model_interaction, pial_left)

slm1 = t_test(slm_basic, -age)
slm2 = t_test(slm_basic, age)

P1, _, _, _ = random_field_theory(slm1)
P2, _, _, _ = random_field_theory(slm2)
print(np.logical_or(P1["P"] < alpha / 2, P2["P"] < alpha / 2))


###################################################################
# Now, imagine that instead of using a fixed effects model, you would prefer a
# mixed effects model wherein handedness is a random variable. This is simple to
# set up. All you need to do is initialize the handedness term with the Random
# class instead, all other code remains identical.


###################################################################

from brainstat.stats.terms import Random

random_handedness = Random(tutorial_data["demographics"]["HAND"], name_ran="Handedness")
random_identity = Random(1, name_ran="identity")
model_random = (
    term_intercept
    + term_age
    + term_iq
    + term_age * term_iq
    + random_handedness
    + random_identity
)
slm_random = linear_model(Y, model_random, pial_left)
slm_random = t_test(slm_random, -age)
P3, _, _, _ = random_field_theory(slm_random)
print(P3)