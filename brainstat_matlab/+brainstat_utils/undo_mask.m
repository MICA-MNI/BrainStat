function Y2 = undo_mask(Y, mask, varargin)
% UNDO_MASK     Restores masked data to its original dimensions. Y2 =
%   UNDO_MASK(Y, mask) restores masked data Y to its original dimension by
%   adding rows. Data that was masked out is filled with nans.
%   
%   Y2 = UNDO_MASK(Y, mask, 'axis', dim) acts along the specified dimension.
%   Defaults to 1. 
%
%   Y2 = UNDO_MASK(Y, mask, 'missing_value', val) fills missing values with 
%   the specified values. Defaults to nan.

p = inputParser;
p.addParameter('axis', 1, @isscalar);
p.addParameter('missing_value', nan, @isscalar);
p.parse(varargin{:});

new_dims = size(Y);
new_dims(p.Results.axis) = numel(mask);
Y2 = ones(new_dims) * p.Results.missing_value;

S.subs = repmat({':'},1,ndims(Y2));
S.subs{p.Results.axis} = mask; 
S.type = '()';
Y2 = subsasgn(Y2,S,Y);
end