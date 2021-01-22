"""
Tutorial 01: Linear Models
=========================================

In this tutorial you will set up your first linear model with SurfStat. Please
note that brainstat does not include sample data yet. Once we introduce
example data loader functions into BrainStat this tutorial will be modified 
accordingly. Until such time, we will rely on randomly generated data and sample
data from BrainSpace.

Setting up a fixed effects model
---------------------------------
Lets start setting up a basic model. Recall that fixed linear models take the
form of :math:`Y = \\beta_0 + \\beta_1x_1 + ... + \\beta_nx_n + \\varepsilon` where
:math:`\\beta_0` denotes the intercept, :math:`\\beta_i` denotes the slope for
variable :math:`x_i`, and :math:`\\varepsilon` is the error term. In BrainStat we
can easily set up such a model as follows.

First lets generate some random data to play with. We'll set up a model with age
and sex as explanatory variables, where age falls in the range 18-35 and sex is 
either 0 or 1.  
"""

###################################################################


import numpy as np

subjects = 20
age = np.random.randint(low=18, high=36, size=(subjects))
sex = np.random.randint(low=0, high=2, size=(subjects)) 


###################################################################
# Next, we can create a BrainStat linear model by declaring these variables as
# terms. The term class requires two things: 1) an array or scalar, and 2) a
# variable name for each column. Lastly, we can create the model by simply
# adding the terms together.


###################################################################


from brainstat.stats.terms import Term
term_intercept = Term(1, names='intercept')
term_age = Term(age, 'age')
term_sex = Term(sex, 'sex')
model = term_intercept + term_age + term_sex


###################################################################
# We can also add interaction effects to the model by multiplying terms.


###################################################################


model_interaction = term_intercept + term_age + term_sex + term_age * term_sex


###################################################################
# Now, lets imagine we have some cortical marker (e.g. cortical thickness) for
# each subject and we want to evaluate whether this marker changes with age
# whilst correcting for effects of sex and age-sex interactions. We could do so
# as follows:


###################################################################


from brainspace.datasets import load_conte69
from brainstat.stats.models import linear_model, t_test

surf_lh, _ = load_conte69()
Y = np.random.rand(subjects, 32492) # Surface has 32492 vertices.
slm = linear_model(Y, model_interaction, surf_lh)
slm = t_test(slm, age)
print(slm['t']) # These are the t-values of the model.


###################################################################
# Never forget: with great models come great multiple comparisons corrections.
# BrainStat provides two methods for these corrections: FDR and random field theory.
# In this example we'll show you how to use random field theory.


###################################################################

from brainstat.stats.multiple_comparisons import random_field_theory
P, _, _, _ = random_field_theory(slm)
