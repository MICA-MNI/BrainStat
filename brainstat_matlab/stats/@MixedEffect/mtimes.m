function s=mtimes(m1,m2)

if (~isa(m1,'FixedEffect') && ~isa(m1,'MixedEffect') && numel(m1)>1) || ...
   (~isa(m2,'FixedEffect') && ~isa(m2,'MixedEffect') && numel(m2)>1)
    warning('If you don''t convert vectors to terms you can get unexpected results :-(') 
end
if ~isa(m1,'MixedEffect')
    m1=MixedEffect([], m1, 'name_fix', inputname(1), 'add_identity', false, 'add_intercept', false);
end
if ~isa(m2,'MixedEffect')
    m2=MixedEffect([], m2, 'name_fix', inputname(2), 'add_identity', false, 'add_intercept', false);
end

if size(m1,3)==1
    v=eye(max(size(m2,1),sqrt(size(m2,3))));   
    m1.variance=FixedEffect(v(:),'I', false);
end
if size(m2,3)==1
    v=eye(max(size(m1,1),sqrt(size(m1,3))));
    m2.variance=FixedEffect(v(:),'I', false);
end

mean=m1.mean*m2.mean;
variance=m1.variance*m2.variance;

N=char(m1.mean);
if ~isempty(N)
    X=double(m1.mean);
    X=X./repmat(max(abs(X)),size(X,1),1);
    k=length(N);
    t=term;
    for i=1:k
        for j=1:i
            if i==j
                v=X(:,i)*X(:,i)';
                t=t+FixedEffect(v(:),N{i}, false);
            else
                v=(X(:,i)+X(:,j))*(X(:,i)+X(:,j))'/4;
                t=t+FixedEffect(v(:),['(' N{j} '+' N{i} ')'], false);
                v=(X(:,i)-X(:,j))*(X(:,i)-X(:,j))'/4;
                t=t+FixedEffect(v(:),['(' N{j} '-' N{i} ')'], false);
            end
        end
    end
    variance = variance+t*m2.variance;
end

N=char(m2.mean);
if ~isempty(N)
    X=double(m2.mean);
    X=X./repmat(max(abs(X)),size(X,1),1);
    k=length(N);
    t=term;
    for i=1:k;
        for j=1:i;
            if i==j
                v=X(:,i)*X(:,i)';
                t=t+FixedEffect(v(:),N{i});
            else
                v=(X(:,i)+X(:,j))*(X(:,i)+X(:,j))'/4;
                t=t+FixedEffect(v(:),['(' N{j} '+' N{i} ')'], false);
                v=(X(:,i)-X(:,j))*(X(:,i)-X(:,j))'/4;
                t=t+FixedEffect(v(:),['(' N{j} '-' N{i} ')'], false);
            end
        end
    end
    variance = variance + m1.variance*t;
end

s = MixedEffect(variance,mean, 'ranisvar', true, 'add_identity', false, 'add_intercept', false);
s = s.set_identity_last();

return
end
