function mustBeSurfStatLattice(surface)
% Validator function for checking SurfStat lattice.
import brainstat_utils.validators.mustContainField
import brainstat_utils.validators.mustBeNDimensional

mustContainField(surface, 'lat');
mustBeA(surface.lat, 'logical');
mustBeNDimensional(surface.lat, 3);