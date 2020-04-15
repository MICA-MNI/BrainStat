function out = plus(t1,t2,precision)
% Combines two effects into a single effect. Discards colinear effects. 
%
% To-do: SurfStat only removes positive colinearity, should we also remove
% negative colinearity i.e. r = -1?

% Check input.
if ~isa(t2,'effect')
    error('Input should consist of effects.');
end

% Extract matrices
x1 = t1.data;
x2 = t2.data;

% Deal with scalar input.
if isscalar(x1)
    x1 = repmat(x1,size(x2));
elseif isscalar(t2)
    x2 = repmat(x2,size(x1));
end

if nargin < 3
    precision = 0.999; 
end

% Check for identical columns (within machine precision), and remove from
% t2. 
r = corr(x1,x2); 
remove = any(r>precision,1); 
if any(remove)
    warning('Detected colinear effects. Removing all but one of each colinear set.');
end

% Build a new effect from the output. 
out = effect([t1.data,t2.data(:,~remove)], ...
           'names',[t1.names;t2.names(~remove)], ...
           'type',t1.type);

end
