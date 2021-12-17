function t_test(obj)


[n,p]=size(obj.X);
obj.contrast=double(obj.contrast);
pinvX=pinv(obj.X);
if length(obj.contrast)<=p
    c=[obj.contrast zeros(1,p-length(obj.contrast))]';
    if sum((null(obj.X)'*c).^2)/sum(c.^2)>eps
        error('Contrast is not estimable :-(');
        return
    end
else
    c=pinvX*obj.contrast;
    r=obj.contrast-obj.X*c;
    if sum(r(:).^2)/sum(obj.contrast(:).^2)>eps
        warning('Contrast is not in the model :-(');
    end
end
obj.c=c';

obj.df=obj.df(length(obj.df));
if ndims(obj.coef)==2
    obj.k=1;
    if isempty(obj.r)
%% fixed effect
        if ~isempty(obj.V)
            Vmh=inv(chol(obj.V)');
            pinvX=pinv(Vmh*obj.X);
        end
        Vc=sum((c'*pinvX).^2,2);
    else
%% mixed effect
        [q1,v]=size(obj.r);
        q=q1+1;
        
        nc=size(obj.dr,2);
        chunk=ceil(v/nc);
        irs=zeros(q1,v);
        for ic=1:nc
            v1=1+(ic-1)*chunk;
            v2=min(v1+chunk-1,v);
            vc=v2-v1+1;
            irs(:,v1:v2)=round(obj.r(:,v1:v2).*repmat(1./obj.dr(:,ic),1,vc));
        end
        [ur, ~, jr]=unique(irs','rows');
        nr=size(ur,1);
        obj.dfs=zeros(1,v);
        Vc=zeros(1,v);
        for ir=1:nr
            iv=(jr==ir);
            rv=mean(obj.r(:,iv),2);
            V=(1-sum(rv))*obj.V(:,:,q);
            for j=1:q1
                V=V+rv(j)*obj.V(:,:,j);
            end
            Vinv=inv(V);
            VinvX=Vinv*obj.X;
            Vbeta=pinv(obj.X'*VinvX);
            G=Vbeta*(VinvX');
            Gc=G'*c;
            R=Vinv-VinvX*G;
            E=zeros(q,1);
            for j=1:q
                E(j)=Gc'*obj.V(:,:,j)*Gc;
                RVV(:,:,j)=R*obj.V(:,:,j);
            end
            for j1=1:q
                for j2=j1:q
                    M(j1,j2)=sum(sum(RVV(:,:,j1).*(RVV(:,:,j2)')));
                    M(j2,j1)=M(j1,j2);
                end
            end
            vc=c'*Vbeta*c;
            Vc(iv)=vc;
            obj.dfs(iv)=vc^2/(E'*pinv(M)*E);
        end
    end
    obj.ef=c'*obj.coef;
    obj.sd=sqrt(Vc.*obj.SSE/obj.df);
    obj.t=obj.ef./(obj.sd+(obj.sd<=0)).*(obj.sd>0);
else
%% multivariate    
    [p,v,k]=size(obj.coef);
    obj.k=k;
    obj.ef=zeros(k,v);
    for j=1:k
        obj.ef(j,:)=c'*obj.coef(:,:,j);
    end
    j=1:k;
    jj=j.*(j+1)/2;
    vf=sum((c'*pinvX).^2,2)/obj.df;
    obj.sd=sqrt(vf*obj.SSE(jj,:));
    
    % Initialize a bunch of stuff to increase performance.
    % If you're reading the following, I'm sorry.
    % Optimization isn't always the most legible.
    % Suffice to say initializing matrices is slow. - RV
    obj.t = zeros(1, size(obj.SSE,2));
    
    upper_tri = triu(ones(obj.k, 'logical'));
    sse_indices = zeros(obj.k);
    sse_indices(upper_tri) = 1:sum(upper_tri(:));
    sse_indices = sse_indices + triu(sse_indices, 1)';

    M = zeros(obj.k+1);

    M_ef = zeros(obj.k+1, 'logical');
    M_ef(2:end, 1) = true;
    M_ef(1, 2:end) = true;

    M_sse = zeros(obj.k+1, 'logical');
    M_sse(2:end, 2:end) = true; 

    ef_duplicate = [obj.ef; obj.ef]; % Trust me - this provides a significant speed boost.
    for ii = 1:size(obj.SSE, 2)
        sse_vertex = obj.SSE(:,ii);
        sse_matrix = sse_vertex(sse_indices);
        det_sse = det(sse_matrix);
        if det_sse <=0
            obj.t(ii) = 0;
        else
            M(M_ef) = ef_duplicate(:,ii);
            M(M_sse) = sse_matrix; 

            obj.t(ii) = sqrt(-det(M) / det_sse / vf);
        end
    end
end
end



