.. _matlab_tutorial1:

Tutorial 1
----------


In this tutorial you will set up your first linear model with BrainStat. First
lets load some example data to play around with. We’ll load age, IQ, and
cortical thickness for a few subjects. Please note that we use the BrainSpace
toolbox to load cortical surfaces.

.. code-block:: matlab

   surfaces = fetch_surfaces();
   [cortical_thickness, demographic_data] = fetch_tutorial_data('n_subjects', 20);

We can create a BrainStat linear model by declaring these variables as terms.
Next, we can create the model by simply adding the terms together. 

.. code-block:: matlab

   term_age = FixedEffect(demographic_data.AGE); 
   term_iq = FixedEffect(demographic_data.IQ); 
   model = term_age + term_iq;

We can also add an interaction effect to the model by multiplying age and handedness.

.. code-block:: matlab

   model_interaction = term_age + term_iq + ...
      term_age * term_iq;

Now, lets imagine we have some cortical marker (e.g. cortical thickness) for
each subject and we want to evaluate whether this marker changes with age whilst
correcting for effects of IQ and age-IQ interactions. 

.. code-block:: matlab

   BrainStatModel = SLM(model_interaction, -demographic_data.AGE, ...
      'surf', surfaces, 'correction', 'rft');
   BrainStatModel.fit(cortical_thickness);

We can access the outcome of this model through its properties e.g.:

.. code-block:: matlab

   BrainStatModel.t; % The t-values. 
   BrainStatModel.P.pval.P; % The vertexwise p-values.

By default BrainStat uses a two-tailed test. If you want to get a one-tailed
test, simply specify it in the SLM model as follows:

.. code-block:: matlab

   BrainStatModel_twotailed = SLM(model_interaction, -demographic_data.AGE, ...
      'correction', 'rft', 'surf', surfaces, 'two_tailed', false);
   BrainStatModel_twotailed.fit(cortical_thickness);

Now, imagine that instead of using a fixed effects model, you would prefer a
mixed effects model wherein handedness is a random variable. This is simple to
set up. All you need to do is initialize the handedness term with the random
class instead, all other code remains identical.

.. code-block:: matlab

   random_handedness = MixedEffect(demographic_data.HAND == "L");
   model_random = term_age + term_iq + term_age * term_iq ...
      + random_handedness;
   
   BrainStatModel_random = SLM(model_random, -demographic_data.AGE, ...
      'surf', surfaces);
   BrainStatModel_random.fit(cortical_thickness)

That concludes the basic usage of the BrainStat for statistical models.
