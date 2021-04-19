function linear_model(obj, Y)

n_samples = size(Y,1);
[obj.X, obj.V] = get_design_matrix(obj, n_samples);
check_error_term(obj.X);

obj.df = n_samples - rank(obj.X);
[residuals, obj.coef, obj.SSE, obj.r, obj.dr] = run_linear_model(obj, Y);

if ~isempty(fieldnames(obj.surf))
    obj.resl = compute_resls(obj, residuals);
end
end

function [X, V] = get_design_matrix(obj, n_samples)

if isa(obj.model, 'random')
    [X, V] = set_mixed_design(obj);
else
    X = set_fixed_design(obj, n_samples);
    V = [];
end
end

function [X, V] = set_mixed_design(obj)
[X, V] = double(obj.model);
[~, q] = size(V);
n = size(X,1);

II = reshape(eye(n),n^2,1);
r = II-V*(pinv(V)*II);
if mean(r.^2)>eps
    warning('Did you forget an error term, I? :-)');
end
if q>1 | ((q==1) & sum(abs(II-V))>0)
    V=reshape(V,[n n q]);
end
end

function X = set_fixed_design(obj, n_samples)
if isa(obj.model, 'term')
    X=double(obj.model);
else
    if prod(size(obj.model))>1
        warning('If you don''t convert vectors to terms you can get unexpected results :-(')
    end
    X = obj.model;
end
if size(X,1)==1
    X=repmat(obj.X, n_samples, 1);
end
end

function check_error_term(X)
r = 1 - X * sum(pinv(X),2);
if mean(r.^2)>eps
    warning('Did you forget a constant term? :-)');
end
end

function [residuals, coef, SSE, r, dr] = run_linear_model(obj, Y)
n_random_effects = get_n_random_effects(obj);
r = [];
dr = [];
if size(Y,3) == 1 % Univariate
    if n_random_effects == 1 % Fixed effects
        [residuals, coef, SSE] = model_univariate_fixed_effects(obj, Y);
    else % Mixed effects
        [residuals, coef, SSE, r, dr] = model_univariate_mixed_effects(obj, Y);
    end
else
    if n_random_effects > 1
        error('Multivariate mixed effects models not implemented.');
    end
    [residuals, coef, SSE] = model_multivariate_fixed_effects(obj, Y);
end
end

function [residuals, coef, SSE] = model_univariate_fixed_effects(obj, Y)
if isempty(obj.V)
    coef=pinv(obj.X)*Y;
    residuals=Y-obj.X*coef;
else
    obj.V=obj.V/mean(diag(obj.V));
    Vmh=inv(chol(obj.V)');
    
    coef=(pinv(Vmh*obj.X)*Vmh)*Y;
    residuals=Vmh*Y-(Vmh*obj.X)*coef;
end
SSE=sum(residuals.^2);
end

function [residuals, coef, SSE, r, dr] = model_univariate_mixed_effects(obj, Y)
n_samples = size(Y,1);
n_vertices = size(Y,2);
n_predictors = size(obj.X,2);
n_random_effects = get_n_random_effects(obj);

q1=n_random_effects-1;
for j=1:n_random_effects
    obj.V(:,:,j)=obj.V(:,:,j)/mean(diag(obj.V(:,:,j)));
end

coef=zeros(n_predictors,n_vertices);
SSE=zeros(1,n_vertices);
obj.r=zeros(q1, n_vertices); % This line could be removed - its a remnant of processing by chunk from SurfStat. 

%% start Fisher scoring algorithm
E=zeros(n_random_effects,n_vertices);
RVV=zeros([n_samples n_samples n_random_effects]);
R=eye(n_samples)-obj.X*pinv(obj.X);
for j=1:n_random_effects
    RV=R*obj.V(:,:,j);
    E(j,:)=sum(Y.*((RV*R)*Y));
    RVV(:,:,j)=RV;
end
M=zeros(n_random_effects);
for j1=1:n_random_effects
    for j2=j1:n_random_effects
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
    irs=round(r.*repmat(1./dr,1,n_vertices));
    [ur,ir,jr]=unique(irs','rows');
    nr=size(ur,1);
    for ir=1:nr
        iv=(jr==ir);
        rv=mean(r(:,iv),2);
        V=(1-sum(rv))*obj.V(:,:,n_random_effects);
        for j=1:q1
            V=V+rv(j)*obj.V(:,:,j);
        end
        Vinv=inv(V);
        VinvX=Vinv*obj.X;
        G=pinv(transpose(obj.X)*VinvX)*(transpose(VinvX));
        R=Vinv-VinvX*G;
        E=zeros(n_random_effects,sum(iv));
        for j=1:n_random_effects
            RV=R*obj.V(:,:,j);
            E(j,:)=sum(Y(:,iv).*((RV*R)*Y(:,iv)));
            RVV(:,:,j)=RV;
        end
        for j1=1:n_random_effects
            for j2=j1:n_random_effects
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
irs=round(r.*repmat(1./dr,1,n_vertices));
[ur,ir,jr]=unique(irs','rows');
nr=size(ur,1);
residuals = Y; 
for ir=1:nr
    iv=(jr==ir);
    rv=mean(r(:,iv),2);
    V=(1-sum(rv))*obj.V(:,:,n_random_effects);
    for j=1:q1
        V=V+rv(j)*obj.V(:,:,j);
    end
    Vmh=inv(chol(V)');
    VmhX=Vmh*obj.X;
    G=pinv(VmhX'*VmhX)*(VmhX')*Vmh;
    coef(:,iv)=G*residuals(:,iv);
    R=Vmh-VmhX*G;
    residuals(:,iv)=R*residuals(:,iv);
    SSE(iv)=sum(residuals(:,iv).^2);
end
end

function [residuals, coef, SSE] = model_multivariate_fixed_effects(obj, Y)
if get_n_random_effects(obj)>1
    error('Multivariate mixed effects models not yet implemented :-(');
end
n_variates = size(Y,3);

if isempty(obj.V)
    X=obj.X;
else
    obj.V=obj.V/mean(diag(obj.V));
    Vmh=inv(chol(V))';
    X=Vmh*obj.X;
    for j=1:n_variates
        Y(:,:,j)=Vmh*Y(:,:,j);
    end
end

%coef=zeros(p,vc,n_variates);
residuals = Y;
for j=1:n_variates
    coef(:,:,j)=pinv(X)*residuals(:,:,j);
    residuals(:,:,j)=residuals(:,:,j)-X*coef(:,:,j);
end
k2=n_variates*(n_variates+1)/2;
SSE=zeros(k2, size(Y,2));
j=0;
for j1=1:n_variates
    for j2=1:j1
        j=j+1;
        SSE(j,:)=sum(residuals(:,:,j1).*residuals(:,:,j2));
    end
end
end

function n_random_effects = get_n_random_effects(obj)
%Gets the number of random effects.

if isa(obj.model, 'random')
    n_random_effects = size(obj.model.variance.matrix, 2);
else
    n_random_effects = 1;
end
end

function resl = compute_resls(obj, residuals)
edg=mesh_edges(obj.surf, obj.mask);

e1=edg(:,1);
e2=edg(:,2);
resl=zeros(size(edg,1), size(residuals,3));

for j=1:size(residuals,3)
    jj=j*(j+1)/2;
    normr=sqrt(obj.SSE(jj,:));
    s=0;
    for i=1:size(residuals,1)
        u=residuals(i,:,j)./normr;  
        s=s+(u(e1)-u(e2)).^2;
    end
    resl(:,j)=s;
end
end

