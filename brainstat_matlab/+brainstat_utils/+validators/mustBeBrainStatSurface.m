function mustBeBrainStatSurface(surface)
% Validator function for checking BrainStat surface.
import brainstat_utils.validators.mustBeSurfStatLattice
import brainstat_utils.validators.mustBeSurfStatSurface

mustBeA(surface, 'struct');
% Accept empty surfaces
if isempty(fieldnames(surface))
    return
end

try
    mustBeSurfStatSurface(surface);
catch 
    try
        mustBeSurfStatLattice(surface);
    catch
        error_id = 'BrainStat:notASurface';
        message = 'Surface must be a valid SurfStat surface or lattice.';
        throwAsCaller(MException(error_id, message));
    end
end

