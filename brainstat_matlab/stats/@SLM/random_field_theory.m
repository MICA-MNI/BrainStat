function [ pval, peak, clus, clusid ] = random_field_theory(obj)

%Corrected P-values for vertices and clusters.
%
% Usage: [ pval, peak, clus, clusid ] = 
%             SurfStatP( obj [, mask, [, clusthresh] ] );
%
% obj.t       = l x v matrix of test statistics, v=#vertices; the first
%               row obj.t(1,:) is the test statistic, and the other 
%               rows are used to calculate cluster resels if k>1. See
%               SurfStatF for the precise definition of the extra rows.
% obj.df      = degrees of freedom.
% obj.dfs     = 1 x v vector of optional effective degrees of freedom.
% obj.k       = k=#variates.
% obj.resl    = e x k matrix of sum over observations of squares of
%               differences of normalized residuals along each edge.
% obj.tri     = 3 x t matrix of triangle indices, 1-based, t=#triangles.
% or
% obj.lat     = nx x ny x nz matrix, 1=in, 0=out, [nx,ny,nz]=size(volume). 
% mask        = 1 x v logical vector, 1=inside, 0=outside, v=#vertices, 
%             = ones(1,v), i.e. the whole surface, by default.
% clusthresh = P-value threshold or statistic threshold for 
%                   defining clusters, 0.001 by default.
%
% pval.P      = 1 x v vector of corrected P-values for vertices.
% pval.C      = 1 x v vector of corrected P-values for clusters.
% peak.t      = np x 1 vector of peaks (local maxima).
% peak.vertid = np x 1 vector of vertex id's (1-based).
% peak.clusid = np x 1 vector of cluster id's that contain the peak.
% peak.P      = np x 1 vector of corrected P-values for the peak.
% clus.clusid = nc x 1 vector of cluster id numbers
% clus.nverts = nc x 1 vector of number of vertices in the cluster.
% clus.resels = nc x 1 vector of resels in the cluster.
% clus.P      = nc x 1 vector of corrected P-values for the cluster.
% clusid      =  1 x v vector of cluster id's for each vertex.
%
% Reference: Worsley, K.J., Andermann, M., Koulis, T., MacDonald, D. 
% & Evans, A.C. (1999). Detecting changes in nonisotropic images.
% Human Brain Mapping, 8:98-101.

% RV - initialize output.
pval = struct('P', [], 'C', []);
peak = struct('t', [], 'vertid', [], 'clusid', [], 'P', []);
clus = struct('clusid', [], 'nverts', [], 'resels', [], 'P', []);
clusid = [];


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

if v==1
    pval.P=stat_threshold(0,1,0,df,[10 obj.t(1)],[],[],[],obj.k,[],[],0);
    pval.P=pval.P(2);
%     peak=[];
%     clus=[];
%     clusid=[];
    return
end

if obj.cluster_threshold<1
    thresh=stat_threshold(0,1,0,df,obj.cluster_threshold,[],[],[],obj.k,[],[],0);
else
    thresh=obj.cluster_threshold;
end

[resels,reselspvert,edg]=obj.compute_resels();

N=sum(obj.mask);
if max(obj.t(1,obj.mask))<thresh

    pval.P=stat_threshold(resels,N,1,df,[10 obj.t],[],[],[],obj.k,[],[],0);
    pval.P=pval.P((1:v)+1);
%     peak=[];
%     clus=[];
%     clusid=[];
else
    [peak,clus,clusid]=obj.peak_clus(thresh,reselspvert,edg);
    [pp,clpval]=stat_threshold(resels,N,1,df,...
        [10 peak.t' obj.t(1,:)],thresh,[10; clus.resels],[],obj.k,[],[],0);
    peak.P=pp((1:length(peak.t))+1)';
    pval.P=pp(length(peak.t)+(1:v)+1);
    if obj.k>1
        j=(obj.k-1):-2:0;
        sphere=zeros(1,obj.k);
        sphere(j+1)=exp((j+1)*log(2)+(j/2)*log(pi)+gammaln((obj.k+1)/2)- ...
            gammaln(j+1)-gammaln((obj.k+1-j)/2));
        sphere=sphere.*(4*log(2)).^(-(0:(obj.k-1))/2)/ndf;
        [pp,clpval]=stat_threshold(conv(resels,sphere),Inf,1,df,...
            [],thresh,[10; clus.resels],[],[],[],[],0);
    end
    clus.P=clpval(2:length(clpval));
    pval.C=interp1([0; clus.clusid],[1; clus.P],clusid);        
end
tlim=stat_threshold(resels,N,1,df,[0.5 1],[],[],[],obj.k,[],[],0);
tlim=tlim(2);
pval.P=pval.P.*(obj.t(1,:)>tlim)+(obj.t(1,:)<=tlim);

return
end

