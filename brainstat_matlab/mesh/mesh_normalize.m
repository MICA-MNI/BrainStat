function [Y, Yav] = mesh_normalize(Y, mask, subdiv)
% MESH_NORMALIZE    Normalizes by subtracting the global mean, or dividing by it.
%   Y = MESH_NORMALIZE=(Y) subtracts the global mean from Y.
%
%   Y = MESH_NORMALIZE(Y, mask) removes vertices outside the logical mask before
%   computing the global mean.
%  
%   Y = MESH_NORMALIZE(Y, mask, subdiv) if subdiv=='s', subtract the global
%   mean. If it's 'd', divide by the global mean.
%
%   [Y, Yav] = MESH_NORMALIZE(Y, ...) also returns the global mean.

if isnumeric(Y)
    [n,v,k]=size(Y);
else
    Ym=Y;
    s=Ym.Format{2};
    if length(s)==2
        s=s([2 1]);
        k=1;
    else
        s=s([3 1 2]);
        k=s(3);
    end
    n=s(1);
    v=s(2);
end    
    
if nargin<2 | isempty(mask)
    mask=logical(ones(1,v));
end
if nargin<3
    subdiv = 's';
end

if isnumeric(Y)
    Yav=squeeze(mean(double(Y(:,mask,:)),2));
    for i=1:n
        for j=1:k
            if subdiv(1) == 's'
                Y(i,:,j)=Y(i,:,j)-Yav(i,j);
            else
                Y(i,:,j)=Y(i,:,j)/Yav(i,j);
            end
        end
    end
else
    Yav=zeros(n,k);
    for i=1:n
        for j=1:k
            if length(s)==2
                Y=Ym.Data(1).Data(:,i);
            else
                Y=Ym.Data(1).Data(:,j,i);
            end
            Yav(i,j)=mean(double(Y));
            if subdiv(1) == 's'
                Y=Y-Yav(i,j);
            else
                Y=Y/Yav(i,j);
            end
            if length(s)==2
                Ym.Data(1).Data(:,i)=Y;
            else
                Ym.Data(1).Data(:,j,i)=Y;
            end            
        end
    end
    Y=Ym;
end

return
end
    
    
    

