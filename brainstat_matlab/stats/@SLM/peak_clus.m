function [ peak, clus, clusid ] = peak_clus(obj, thresh, ...
    reselspvert, edg)

%Finds peaks (local maxima) and clusters for surface data.
% 
% Usage: [ peak, clus, clusid ] = SurfStatPeakClus( obj, mask, thresh ...
%                                    [, reselspvert [, edg ] ] );
%
% obj.t       = l x v matrix of data, v=#vertices; the first row
%               obj.t(1,:) is used for the clusters, and the other 
%               rows are used to calculate cluster resels if k>1. See
%               SurfStatF for the precise definition of the extra rows.
% obj.tri     = t x 3 matrix of triangle indices, 1-based, t=#triangles.
% or
% obj.lat     = nx x ny x nz matrix, 1=in, 0=out, [nx,ny,nz]=size(volume). 
% mask        = 1 x v vector, 1=inside, 0=outside, v=#vertices.
% thresh      = clusters are vertices where obj.t(1,mask)>=thresh.
% reselspvert = 1 x v vector of resels per vertex, default ones(1,v).
% edg         = e x 2 matrix of edge indices, 1-based, e=#edges.
% The following are optional:
% obj.df      = degrees of freedom - only the length (1 or 2) is used
%               to determine if obj.t is Hotelling's T or T^2 when k>1.
% obj.k       = k=#variates, 1 by default.
%
% peak.t      = np x 1 vector of peaks (local maxima).
% peak.vertid = np x 1 vector of vertex id's (1-based).
% peak.clusid = np x 1 vector of cluster id's that contain the peak.
% clus.clusid = nc x 1 vector of cluster id numbers
% clus.nverts = nc x 1 vector of number of vertices in the cluster.
% clus.resels = nc x 1 vector of resels in the cluster.
% clusid      =  1 x v vector of cluster id's for each vertex.

if ~exist('edg','var')
    edg=mesh_edges(obj.surf);
end

[l,v]=size(obj.t);

local_t = obj.t;
local_t(1,~obj.mask)=min(obj.t(1,:));
t1=local_t(1,edg(:,1));
t2=local_t(1,edg(:,2));
iobj=ones(1,v);
iobj(edg(t1<t2,1))=0;
iobj(edg(t2<t1,2))=0;
lmvox=find(iobj);
lmt=local_t(1,lmvox);

excurset=(local_t(1,:)>=thresh);
n=sum(excurset);

if n<1
   peak=[];
   clus=[];
   clusid=[];
   return
end

voxid=cumsum(excurset);
edg=voxid(edg(all(excurset(edg),2),:));

% Find cluster id's in nf (from Numerical Recipes in C, page 346):
nf=1:n;
for el=1:size(edg,1)
    j=edg(el,1);
    k=edg(el,2);
    while nf(j)~=j j=nf(j); end
    while nf(k)~=k k=nf(k); end
    if j~=k nf(j)=k; end
end
for j=1:n
    while nf(j)~=nf(nf(j)) nf(j)=nf(nf(j)); end
end

% find the unique cluster id's corresponding to the local maxima:
vox=find(excurset);
ivox=find(ismember(vox,lmvox));
clmid=nf(ivox);
[uclmid,iclmid,jclmid]=unique(clmid);

% find their volumes:
ucid=unique(nf);
nclus=length(ucid);
ucvol=histc(nf,ucid); 

% find their resels:
if ~exist('reselspvert', 'var')
    reselsvox=ones(size(vox));
else
    reselsvox=reselspvert(vox);
end
nf1=interp1([0 ucid],0:nclus,nf,'nearest');

% if k>1, find volume of cluster in added sphere:
if isempty(obj.k) | obj.k==1
    ucrsl=accumarray(nf1',reselsvox)';
end
if ~isempty(obj.k) & obj.k==2
    if l==1
        ndf=length(obj.df);
        r=2*acos((thresh./local_t(1,vox)).^(1/ndf));
    else
        r=2*acos(sqrt((thresh-local_t(2,vox)).*(thresh>=local_t(2,vox))./ ...
            (local_t(1,vox)-local_t(2,vox))));
    end
    ucrsl=accumarray(nf1',r'.*reselsvox')';
end
if ~isempty(obj.k) & obj.k==3
    if l==1
        ndf=length(obj.df);
        r=2*pi*(1-(thresh./local_t(1,vox)).^(1/ndf));
    else
        nt=20;
        theta=((1:nt)'-1/2)/nt*pi/2;
        s=(cos(theta).^2)*local_t(2,vox);
        if l==3
            s=s+(sin(theta).^2)*local_t(3,vox);
        end
        r=2*pi*mean(1-sqrt((thresh-s).*(thresh>=s)./ ...
            (ones(nt,1)*local_t(1,vox)-s)));
    end
    ucrsl=accumarray(nf1',r'.*reselsvox')';
end

% and their ranks (in ascending order):
[sortucrsl,iucrsl]=sort(ucrsl);
rankrsl=zeros(1,nclus);
rankrsl(iucrsl)=nclus:-1:1;

% add these to lm as extra columns:
lmid=lmvox(ismember(lmvox,vox));

% The line lm=... bugs out when there is only one cluster as
% rankrsl(jclmid) becomes a column vector instead of a row vector.
% Fix by enforcing the vector directionality - RV.
% lm=flipud(sortrows([local_t(1,lmid)' lmid' rankrsl(jclmid)'],1));
lm = flipud(sortrows([ ...
    column_vector(local_t(1, lmid)), ...
    column_vector(lmid), ...
    column_vector(rankrsl(jclmid))], ...
    1));

cl=sortrows([rankrsl' ucvol' ucrsl'],1);

clusid=zeros(1,v);
clusid(vox)=interp1([0 ucid],[0 rankrsl],nf,'nearest');

peak.t=lm(:,1);
peak.vertid=lm(:,2);
peak.clusid=lm(:,3);
clus.clusid=cl(:,1);
clus.nverts=cl(:,2);
clus.resels=cl(:,3);

return
end

function v_col = column_vector(v)
% Converts the input vector to a column vector.
arguments
    v {mustBeVector}
end
v_col = v(:);
end
