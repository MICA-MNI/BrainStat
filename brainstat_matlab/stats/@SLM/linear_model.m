function linear_model(obj, Y)
%%TODO :: Add mask change from Python.

%Fits linear mixed effects models to surface data and estimates resels.
%
% Usage: obj = SurfStatLinMod( Y, M [,surf [,niter [,thetalim [,drlim]]]]);
%
% Y        = n x v or n x v x k matrix of surface data, v=#vertices;
%            n=#observations; k=#variates, or memory map of same.      
% M        = model formula of class term or random; or scalar or
%            n x p design matrix of p regressors for the linear model.
% surf.tri = t x 3 matrix of triangle indices, 1-based, t=#triangles.
% or
% surf.lat = nx x ny x nz matrix, 1=in, 0=out, [nx,ny,nz]=size(volume). 
% The following arguments are not usually used:
% niter    = number of extra iterations of the Fisher scoring algorithm
%            for fitting mixed effects models. Default 1.
% thetalim = lower limit on variance coefficients, in sd's. Default 0.01.
% drlim    = step of ratio of variance coefficients, in sd's. Default 0.1. 
%
% obj.X    = n x p design matrix.
% obj.V    = n x n x q variance matrix bases, normalised so that
%            mean(diag(obj.V))=1. Absent if q=1 and obj.V=eye(n).
% obj.df   = degrees of freedom = n-rank(X).
% obj.coef = p x v x k matrix of coefficients of the linear model.
% obj.SSE  = k*(k+1)/2 x v matrix of sum of squares of errors
%            whitened by obj.V*obj.r.
% obj.r    = (q-1) x v matrix of coefficients of the first (q-1)
%            components of obj.V divided by their sum. 
%            Coefficients are clamped to a minimum of 0.01 x sd.
% obj.dr   = (q-1) x nc vector of increments in obj.r = 0.1 x sd.
% If surf is not empty, returns:
% obj.resl = e x k matrix of sum over observations of squares of 
%            differences of normalized residuals along each edge.
% obj.tri  = surf.tri,
% or
% obj.lat  = surf.lat.

maxchunk=2^20;

if isnumeric(Y)
    s=size(Y);
    isnum=true;
else
    Ym=Y;
    s=Ym.Format{2};
    if length(s)==2
        s=s([2 1]);
    else
        s=s([3 1 2]);
    end
    [PATHSTR,NAME,EXT]=fileparts(Ym.Filename);
    filenameResid=fullfile(PATHSTR,['Resid.mmm']);
    copyfile(Ym.Filename,filenameResid);
    YmResid=memmapfile(filenameResid,'Format',{'single' s 'Resid'},'Writable',true);
    isnum=false;
end
n=s(1);
v=s(2);
if length(s)==2
    k=1;
else
    k=s(3);
end

if isa(obj.model, 'random')
    [obj.X,V]=double(obj.model);
    [n2 q]=size(V);
    II=reshape(eye(n),n^2,1);
    r=II-V*(pinv(V)*II);
    if mean(r.^2)>eps
        warning('Did you forget an error term, I? :-)');
    end
    if q>1 | ((q==1) & sum(abs(II-V))>0)
        obj.V=reshape(V,[n n q]);
    end
    clear V II r;
else
    q=1;
    if isa(obj.model, 'term')
        obj.X=double(obj.model);
    else
        if prod(size(obj.model))>1
            warning('If you don''t convert vectors to terms you can get unexpected results :-(')
        end
        obj.X = obj.model;
    end
    if size(obj.X,1)==1
        obj.X=repmat(obj.X,n,1);
    end
end

pinvX=pinv(obj.X);
r=ones(n,1)-obj.X*(pinvX*ones(n,1));
if mean(r.^2)>eps
    warning('Did you forget a constant term? :-)');
end

p=size(obj.X,2);
obj.df=n-rank(obj.X);
obj.coef=zeros(p,v,k);
k2=k*(k+1)/2;
obj.SSE=zeros(k2,v);

if isnum
    nc=1;
    chunk=v;
else
    nc=ceil(v*n*k/maxchunk);
    chunk=ceil(v/nc);
end
if ~isnum
    fprintf(1,'%s',[num2str(round(v*n*k*4/2^20)) ' Mb to fit, % remaining: 100 ']);
    n10=floor(n/10);
end
for ic=1:nc
    if ~isnum & rem(ic,n10)==0 
        fprintf(1,'%s',[num2str(round(100*(1-ic/nc))) ' ']);
    end
    v1=1+(ic-1)*chunk;
    v2=min(v1+chunk-1,v);
    vc=v2-v1+1;
    if isnum       
        Y=double(Y);
    else
        if length(s)==2
            Y=double(Ym.Data(1).Data(v1:v2,:)');
        else
            Y=double(permute(Ym.Data(1).Data(v1:v2,:,:),[3 1 2]));
        end
    end
    if k==1
        if q==1
            %% fixed effects
            if ~isfield(obj,'V')
                coef=pinvX*Y;
                Y=Y-obj.X*coef;
            else
                if ic==1
                    obj.V=obj.V/mean(diag(obj.V));
                    Vmh=inv(chol(obj.V)');
                end
                coef=(pinv(Vmh*obj.X)*Vmh)*Y;
                Y=Vmh*Y-(Vmh*obj.X)*coef;
            end
            SSE=sum(Y.^2);
        else
            %% mixed effects
            if ic==1
                q1=q-1;
                for j=1:q
                    obj.V(:,:,j)=obj.V(:,:,j)/mean(diag(obj.V(:,:,j)));;
                end
                obj.r=zeros(q1,v);
                obj.dr=zeros(q1,nc);
            end
            coef=zeros(p,vc);
            SSE=zeros(1,vc);

            %% start Fisher scoring algorithm
            E=zeros(q,vc);
            RVV=zeros([n n q]);
            R=eye(n)-obj.X*pinv(obj.X);
            for j=1:q
                RV=R*obj.V(:,:,j);
                E(j,:)=sum(Y.*((RV*R)*Y));
                RVV(:,:,j)=RV;
            end
            M=zeros(q);
            for j1=1:q
                for j2=j1:q
                    M(j1,j2)=sum(sum(RVV(:,:,j1).*(RVV(:,:,j2)')));
                    M(j2,j1)=M(j1,j2);
                end
            end
            theta=pinv(M)*E;

            tlim=sqrt(2*diag(pinv(M)))*sum(theta)*obj.thetalim;
            theta=theta.*(theta>=tlim)+tlim.*(theta<tlim);
            r=theta(1:q1,:)./repmat(sum(theta),q1,1);

            Vt=2*pinv(M);
            m1=diag(Vt);
            m2=2*sum(Vt)';
            Vr=m1(1:q1)-m2(1:q1).*mean(obj.r,2)+sum(Vt(:))*mean(r.^2,2);
            dr=sqrt(Vr)*obj.drlim;

            %% Exrtra Fisher scoring iterations
            for iter=1:obj.niter
                irs=round(r.*repmat(1./dr,1,vc));
                [ur,ir,jr]=unique(irs','rows');
                nr=size(ur,1);
                for ir=1:nr
                    iv=(jr==ir);
                    rv=mean(r(:,iv),2);
                    V=(1-sum(rv))*obj.V(:,:,q);
                    for j=1:q1
                        V=V+rv(j)*obj.V(:,:,j);
                    end
                    Vinv=inv(V);
                    VinvX=Vinv*obj.X;
                    G=pinv(transpose(obj.X)*VinvX)*(transpose(VinvX));
                    R=Vinv-VinvX*G;
                    E=zeros(q,sum(iv));
                    for j=1:q
                        RV=R*obj.V(:,:,j);
                        E(j,:)=sum(Y(:,iv).*((RV*R)*Y(:,iv)));
                        RVV(:,:,j)=RV;
                    end
                    for j1=1:q
                        for j2=j1:q
                            M(j1,j2)=sum(sum(RVV(:,:,j1).*(RVV(:,:,j2)')));
                            M(j2,j1)=M(j1,j2);
                        end
                    end
                    thetav=pinv(M)*E;
                    tlim=sqrt(2*diag(pinv(M)))*sum(thetav)*obj.thetalim;
                    theta(:,iv)=thetav.*(thetav>=tlim)+tlim.*(thetav<tlim);
                end
                r=theta(1:q1,:)./(ones(q1,1)*sum(theta));
            end

            %% finish Fisher scoring
            irs=round(r.*repmat(1./dr,1,vc));
            [ur,ir,jr]=unique(irs','rows');
            nr=size(ur,1);
            for ir=1:nr
                iv=(jr==ir);
                rv=mean(r(:,iv),2);
                V=(1-sum(rv))*obj.V(:,:,q);
                for j=1:q1
                    V=V+rv(j)*obj.V(:,:,j);
                end
                Vmh=inv(chol(V)');
                VmhX=Vmh*obj.X;
                G=pinv(VmhX'*VmhX)*(VmhX')*Vmh;
                coef(:,iv)=G*Y(:,iv);
                R=Vmh-VmhX*G;
                Y(:,iv)=R*Y(:,iv);
                SSE(iv)=sum(Y(:,iv).^2);
            end
            obj.r(:,v1:v2)=r;
            obj.dr(:,ic)=dr;
        end
    else
        %% multivariate
        if q>1
            error('Multivariate mixed effects models not yet implemented :-(');
            return
        end

        if ~isfield(obj,'V')
            X=obj.X;
        else
            if ic==1
                obj.V=obj.V/mean(diag(obj.V));
                Vmh=inv(chol(obj.V))';
                X=Vmh*obj.X;
                pinvX=pinv(X);
            end
            for j=1:k
                Y(:,:,j)=Vmh*Y(:,:,j);
            end
        end

        coef=zeros(p,vc,k);
        for j=1:k
            coef(:,:,j)=pinvX*Y(:,:,j);
            Y(:,:,j)=Y(:,:,j)-X*coef(:,:,j);
        end
        k2=k*(k+1)/2;
        SSE=zeros(k2,vc);
        j=0;
        for j1=1:k
            for j2=1:j1
                j=j+1;
                SSE(j,:)=sum(Y(:,:,j1).*Y(:,:,j2));
            end
        end
    end
    obj.coef(:,v1:v2,:)=coef;
    obj.SSE(:,v1:v2)=SSE;
    if ~isnum
        YmResid.Data(1).Resid(:,v1:v2,:)=single(Y);
    end
end
if ~isnum
    fprintf(1,'%s\n','Done');
end

%% resels
if isempty(fieldnames(obj.surf))
    return
end

edg=SurfStatEdg(obj.surf);

e=size(edg,1);
e1=edg(:,1);
e2=edg(:,2);
clear edg
obj.resl=zeros(e,k);
if ~isnum
    fprintf(1,'%s',[num2str(n) ' x ' num2str(k) ' surfaces to resel, % remaining: 100 ']);
    n10=floor(n*k/10);
end
for j=1:k
    jj=j*(j+1)/2;
    normr=sqrt(obj.SSE(jj,:));
    s=0;
    for i=1:n
        if ~isnum & rem((j-1)*n+i,n10)==0
            fprintf(1,'%s',[num2str(round(100*(1-((j-1)*n+i)/(n*k)))) ' ']);
        end
        if isnum
            u=Y(i,:,j)./normr;
        else
            u=double(YmResid.Data(1).Resid(i,:,j))./normr;
        end
        s=s+(u(e1)-u(e2)).^2;
    end
    obj.resl(:,j)=s;
end
if ~isnum
    clear YmResid
    delete(filenameResid);
    fprintf(1,'%s\n','Done');
end

return
end