from brainstat.context.genetics import surface_genetic_expression
from brainstat.datasets import fetch_parcellation, fetch_template_surface, fetch_parcellation
import numpy as np
from nilearn import datasets

# Get Schaefer-100 genetic expression.
# def test_surface_genetic_expression():
#     schaefer_100_fs5 = fetch_parcellation("fsaverage5", "schaefer", 100)
#     surfaces = fetch_template_surface("fsaverage5", join=False)
#     expression = surface_genetic_expression(schaefer_100_fs5, surfaces, space="fsaverage")
#     assert expression is not None


# def test_surface_genetic_expression2():
#     destrieux = datasets.fetch_atlas_surf_destrieux()
#     labels = np.hstack((destrieux['map_left'], destrieux['map_right']))
#     fsaverage = datasets.fetch_surf_fsaverage()
#     surfaces = (fsaverage['pial_left'], fsaverage['pial_right'])
#     expression = surface_genetic_expression(labels, surfaces, space='fsaverage')
#     assert expression is not None