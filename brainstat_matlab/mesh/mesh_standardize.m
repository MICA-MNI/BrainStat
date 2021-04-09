function [ Y, Ym ] = mesh_standardize(Y, mask, subtractordivide);
% MESH_STANDARDIZE    Standardizes by subtracting the global mean, or dividing by it.
%   Y = MESH_STANDARDIZE=(Y) subtracts the global mean from Y.
%
%   Y = MESH_STANDARDIZE(Y, mask) removes vertices outside the logical mask before
%   computing the global mean.
%  
%   Y = MESH_STANDARDIZE(Y, mask, subdiv) if subdiv=='s', subtract the global
%   mean. If it's 'd', divide by the global mean.
%
%   [Y, Yav] = MESH_STANDARDIZE(Y, ...) also returns the global mean.

if nargin<2 | isempty(mask)
    mask=logical(ones(1,size(Y,2)));
end
if nargin<3
    subtractordivide = 's';
end
Ym=mean(Y(:,mask),2);
for i=1:size(Y,1)
    if subtractordivide(1) == 's'
        Y(i,:)=Y(i,:)-Ym(i);
    else
        Y(i,:)=(Y(i,:)/Ym(i)-1)*100;
    end
end

return
end
    
    
    

