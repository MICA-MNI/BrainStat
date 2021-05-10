function Y2 = apply_mask(Y, mask, axis)
% APPLY_MASK    Retains only data within the mask.
%   Y2 = APPLY_MASK(Y, mask) removes columns from Y that are false in the mask.
%   mask must be a vector of length equal to size(Y,2)
%
%   Y2 = APPLY_MASK(Y, mask, axis) applies the mask along the dimensions axis.   

if ~exist('axis','var')
    axis = 1;
end

S.subs = repmat({':'}, 1, ndims(Y));
S.subs{axis} = ~mask;
S.type = '()';
Y2 = subsasgn(Y, S, []);

end