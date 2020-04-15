function s=minus(obj,t2,precision)
% Removes columns occuring in both t1 and t2 from t1, accounting for
% overall scaling of the vectors.

% Check input.
if ~isa(t2,'effect')
    error('Input must be a effect.')
end
if nargin < 3
    precision = 0.999;
end

% Deal with scalar input.
if isscalar(obj.data)
    x1 = repmat(obj.data,size(t2.data));
else
    x1 = obj.data;
end
if isscalar(t2.data)
    x2 = repmat(t2.data,size(obj.data));
else
    x2 = t2.data;
end

% Check for identical columns (within machine precision), and remove from
% t2. 
r = corr(x1,x2); 
remove = sum(r>precision,2); 
   
% Build a new effect from the output. 
s = effect(obj.data(:,~remove));

end

