function s=minus(m1,m2)

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

s = MixedEffect(m1.variance - m2.variance, m1.mean - m2.mean, ...
    'ranisvar', true, 'add_identity', false, 'add_intercept', false);
s = s.set_identity_last();
