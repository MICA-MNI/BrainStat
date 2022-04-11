function m=I
% I   Identity matrix for the variance in a model formula.
%   m = I() is identical to I = MixedEffect(1). Function is retained only
%   for ease of use for SurfStat users. BrainStat adds the identity matrix
%   automatically to MixedEffects i.e., this function should never be needed. 
m=MixedEffect(1);
