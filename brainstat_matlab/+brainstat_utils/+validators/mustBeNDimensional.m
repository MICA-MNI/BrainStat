function mustBeNDimensional(x, n_dimensions)
% MUSTBENDIMENSIONAL    Throws an error if input is incorrectly sized.
%   MUSTBENDIMENSIONAL(x, n_dimensions) throws an error if ndims(x) is not
%   equal to n_dimensions.

if ndims(x) ~= n_dimensions
    erorr_id = 'BrainStat:notNDimensional';
    message = sprintf('Input is %d dimensional, but must be %d dimensional.', ndims(x), n_dimensions);
    throwAsCaller(MException(erorr_id, message));
end
end