function mustBeSurfStatSurface(surface)
% Validator function for checking SurfStat surfaces.
import brainstat_utils.validators.mustContainField
import brainstat_utils.validators.mustBeOfSize

mustContainField(surface, 'tri');
mustContainField(surface, 'coord');
    
mustBeInteger(surface.tri);
mustBePositive(surface.tri);

mustBeNumeric(surface.coord);

mustBeOfSize(surface.tri, [0, 3]);
mustBeOfSize(surface.coord, [3, 0]);