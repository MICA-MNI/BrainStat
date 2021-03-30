function fdr(obj)

%Q-values for False Discovey Rate of resels.
%
% Usage: qval = SurfStatQ( obj [, mask] );
%
% obj.t    = 1 x v vector of test statistics, v=#vertices.
% obj.df   = degrees of freedom.
% obj.dfs  = 1 x v vector of optional effective degrees of freedom.
% obj.k    = #variates.
% mask     = 1 x v logical vector, 1=inside, 0=outside, 
%          = ones(1,v), i.e. the whole surface, by default.
% The following are optional:
% obj.resl = e x k matrix of sum over observations of squares of
%           differences of normalized residuals along each edge.
% obj.tri  = t x 3 matrix of triangle indices, 1-based, t=#triangles.
% or
% obj.lat  = 3D logical array, 1=in, 0=out.
%
% qval      = 1 x v vector of Q-values.

[l,v]=size(obj.t);
if isempty(obj.mask)
    obj.mask=logical(ones(1,v));
end

df=zeros(2);
ndf=length(obj.df);
df(1,1:ndf)=obj.df;
df(2,1:2)=obj.df(ndf);
if ~isempty(obj.dfs)
    df(1,ndf)=mean(obj.dfs(obj.mask>0));
end

if ~isempty(obj.du)
    [resels,reselspvert] = obj.compute_resels();
else
    reselspvert=ones(1,v);
end
reselspvert=reselspvert(obj.mask);

P_val=stat_threshold(0,1,0,df,[10 obj.t(1,obj.mask)],[],[],[],obj.k,[],[],0);
P_val=P_val(2:length(P_val));
np=length(P_val);
[P_sort, index]=sort(P_val);
r_sort=reselspvert(index);
c_sort=cumsum(r_sort);
P_sort=P_sort./(c_sort+(c_sort<=0)).*(c_sort>0)*sum(r_sort);
m=1;
Q_sort=zeros(1,np);
for i=np:-1:1
    if P_sort(i)<m
        m=P_sort(i);
    end
    Q_sort(i)=m;
end
Q=zeros(1,np);
Q(index)=Q_sort;

obj.Q=ones(1,size(obj.mask,2));
obj.Q(obj.mask)=Q;

end

