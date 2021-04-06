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
        [ur,ir,jr]=unique(irs','rows');
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
            iv=(ir==jr);
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
    if k==2
        det=obj.SSE(1,:).*obj.SSE(3,:)-obj.SSE(2,:).^2;
        obj.t=obj.ef(1,:).^2.*obj.SSE(3,:) + ...
              obj.ef(2,:).^2.*obj.SSE(1,:) - ...
              2*obj.ef(1,:).*obj.ef(2,:).*obj.SSE(2,:);
    end
    if k==3
        det=obj.SSE(1,:).*(obj.SSE(3,:).*obj.SSE(6,:)-obj.SSE(5,:).^2) - ...
            obj.SSE(6,:).*obj.SSE(2,:).^2 + ...
            obj.SSE(4,:).*(obj.SSE(2,:).*obj.SSE(5,:)*2-obj.SSE(3,:).*obj.SSE(4,:));
        obj.t=      obj.ef(1,:).^2.*(obj.SSE(3,:).*obj.SSE(6,:)-obj.SSE(5,:).^2);
        obj.t=obj.t+obj.ef(2,:).^2.*(obj.SSE(1,:).*obj.SSE(6,:)-obj.SSE(4,:).^2);
        obj.t=obj.t+obj.ef(3,:).^2.*(obj.SSE(1,:).*obj.SSE(3,:)-obj.SSE(2,:).^2);
        obj.t=obj.t+2*obj.ef(1,:).*obj.ef(2,:).*(obj.SSE(4,:).*obj.SSE(5,:)-obj.SSE(2,:).*obj.SSE(6,:));
        obj.t=obj.t+2*obj.ef(1,:).*obj.ef(3,:).*(obj.SSE(2,:).*obj.SSE(5,:)-obj.SSE(3,:).*obj.SSE(4,:));
        obj.t=obj.t+2*obj.ef(2,:).*obj.ef(3,:).*(obj.SSE(2,:).*obj.SSE(4,:)-obj.SSE(1,:).*obj.SSE(5,:));
    end
    if k>3
         warning('Hotelling''s T for k>3 not programmed yet');
         return
    end
    obj.t=obj.t./(det+(det<=0)).*(det>0)/vf;
    obj.t=sqrt(obj.t+(obj.t<=0)).*(obj.t>0);
end
end



