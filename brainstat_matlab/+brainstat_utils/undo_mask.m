function undo_mask(Y, mask, varargin)
% Restores the original dimensions of masked data.
%
% Comments to be added.

p = inputParser;
p.addParameter('axis', 1, @isscalar);
p.addParameter('missing_value', nan, @isscalar);
p.parse(varargin{:});

new_dims = size(Y);
new_dims(p.Results.axis) = numel(mask);
Y2 = ones(new_dims) * p.Results.missing_value;

% Permute axes so we always run along the first.
permutation = 1:ndims(Y);
permutation([1, mask]) = permutation([mask, 1]);
Y = permute(Y, permutation);
Y2 = permute(Y2, permutation);
Y2(mask, :) = Y;
Y2 = permute(Y2, permutation); 
end