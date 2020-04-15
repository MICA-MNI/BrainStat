function lm = BrainStatLinMod(Y,model,contrast)

% Grab some basic parameters. 
X = model.data; 

% If no intercept, add an intercept.
if ~any(all(X==(X(1,:))))
    warning('Did not find an intercept in the model, adding one now.');
    X = [ones(size(X,1),1),X];
end

[n,p] = size(X); 
df = n-p; 

% Convert the contrast to a beta weighting vector. 
lambda = pinv(X)*contrast; 
% TO-DO: Add an estimatability check (see SurfStatT). 

% Compute the t-values and p-values
if model.type == "fixed"
    % Simple fixed effects linear model.
    beta = (X.' * X)^-1 * X.' * Y;     
    SSE = sum((Y - X * beta).^2);
elseif model.type == "random"
    error('Mixed/Random effects models have not been implemented yet.');
end

% Compute contrast estimate. See also SPM documentation chapter 4
% https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pdf.
contrast_estimate = lambda' * beta; 
Vc = sum((lambda' * pinv(X).^2),2); % what does Vc stand for? I simply copied this variable name from SurfStat.
contrast_var = sqrt(Vc * SSE / df);
t = contrast_estimate ./ contrast_var + (contrast_var <=0).* (contrast_var>0); 

% Assign variables to the output model. 
lm = struct('X',X,'beta',beta,'deg_freedom',df,'t',t,'p',p);



