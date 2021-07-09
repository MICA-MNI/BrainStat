"""
Tutorial 01: Linear Models
=========================================
In this tutorial you will set up your first linear model with BrainStat. 
First lets load some example data to play around with. We'll load age, IQ, and left
hemispheric cortical thickness for a few subjects.
"""

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
mask = np.all(thickness != 0, axis=0)

pial_left = read_surface_gz(fetch_surf_fsaverage()["pial_left"])

###################################################################
# Next, we can create a BrainStat linear model by declaring these variables as
# terms. The term class requires two things: 1) an array or scalar, and 2) a
# variable name for each column. Lastly, we can create the model by simply
# adding the terms together.

from brainstat.stats.terms import FixedEffect

term_age = FixedEffect(age, "age")
term_iq = FixedEffect(iq, "iq")
model = term_age + term_iq

###################################################################
# We can also add interaction effects to the model by multiplying terms.

model_interaction = term_age + term_iq + term_age * term_iq

###################################################################
# Now, lets imagine we have some cortical marker (e.g. cortical thickness) for
# each subject and we want to evaluate whether this marker changes with age
# whilst correcting for effects of sex and age-sex interactions.

from brainstat.stats.SLM import SLM

slm = SLM(model_interaction, -age, surf=pial_left, correction="rft", mask=mask)
slm.fit(thickness)
print(slm.t.shape)  # These are the t-values of the model.
print(slm.P["pval"]["P"])  # These are the random field theory derived p-values.

###################################################################
# By default BrainStat uses a two-tailed test. If you want to get a one-tailed
# test, simply specify it in the SLM model as follows:

slm_two_tails = SLM(
    model_interaction, -age, surf=pial_left, correction="rft", two_tailed=False
)
slm_two_tails.fit(thickness)

###################################################################
# Now, imagine that instead of using a fixed effects model, you would prefer a
# mixed effects model wherein handedness is a random variable. This is simple to
# set up. All you need to do is initialize the handedness term with the MixedEffect
# class instead, all other code remains identical.

from brainstat.stats.terms import MixedEffect

random_handedness = MixedEffect(
    tutorial_data["demographics"]["HAND"], name_ran="Handedness"
)

model_random = term_age + term_iq + term_age * term_iq + random_handedness
slm_random = SLM(model_random, -age, surf=pial_left, correction="fdr", mask=mask)
slm_random.fit(thickness)

###############################################################################
# That concludes the basic usage of the BrainStat for statistical models.
