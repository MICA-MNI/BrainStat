function s=mpower(t,p)
t=FixedEffect(t, inputname(1), false);
if p>=2
    s=mtimes(t,mpower(t,p-1));
else 
    s=t;
end
    
