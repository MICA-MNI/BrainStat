function Y2 = undo_mask(Y, mask, varargin)
% Restores the original dimensions of masked data, adding the missing
% value to all missing data.
%
% Comments to be added.

p = inputParser;
p.addParameter('axis', 1, @isscalar);
p.addParameter('missing_value', nan, @isscalar);
p.parse(varargin{:});

new_dims = size(Y);
new_dims(p.Results.axis) = numel(mask);
Y2 = ones(new_dims) * p.Results.missing_value;

S2.subs = repmat({':'},1,ndims(Y2));
S2.subs{p.Results.axis} = mask; 
S2.type = '()';
Y2 = subsasgn(Y2,S,Y);
end