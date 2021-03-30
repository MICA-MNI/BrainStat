function Y2 = apply_mask(Y, mask, axis)
% Applies a mask along the specified axis. 
if ~exist('axis','var')
    axis = 1;
end

S.subs = repmat({':'}, 1, ndims(Y));
S.subs{axis} = ~mask;
S.type = '()';
Y2 = subsasgn(Y, S, []);

end