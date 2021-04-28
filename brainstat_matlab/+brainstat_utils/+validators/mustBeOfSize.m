function mustBeOfSize(x, dimension_size)
% Validator function for checking number of dimensions. 
% Unlimited dimension sizes can be provided in order. To skip a dimension,
% provide a 0. 

arguments
    x
    dimension_size {mustBeInteger, mustBeNonnegative, mustBeVector} 
end

if ndims(x) ~= numel(dimension_size)
    eid = 'BrainStat:notNDimensional';
    msg = sprintf('Input is %d dimensional, but must be %d dimensional.', ndims(x), n_dimensions);
    throwAsCaller(MException(eid, msg));
end

sz = size(x);
for ii = 1:numel(sz)
    if sz(ii) ~= dimension_size(ii) && dimension_size(ii) ~= 0
        eid = 'BrainStat:notCorrectSize';
        msg = sprintf('Input dimension %d has size %d but should be size %d.', ...
            ii, sz(ii), dimension_size(ii));
        throwAsCaller(MException(eid, msg));
    end
end
