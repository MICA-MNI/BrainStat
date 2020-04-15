function lm = BrainStatLinMod(Y,model,contrast,surf)

%% Compute t-value for the contrast. 
% Grab some basic parameters. 
X = model.data; 

% If no intercept, add an intercept.
if ~any(all(X==(X(1,:))))
    warning('Did not find an intercept in the model, adding one now.');
    X = [ones(size(X,1),1),X];
end

% Degrees o' Freedom
df = size(X,1)-size(X,2); % Could adjust for rank-deficient matrices. 

% Convert the contrast to a beta weighting vector. 
lambda = pinv(X)*contrast; 
% TO-DO: Add an estimatability check (see SurfStatT). 

% Get beta and t values. 
if size(Y,3) > 1
    error('Multivariate models have not been implemented.');
    %[beta,t] = multivariate_model(X,Y,lambda,df); 
elseif model.type == "fixed"
    [beta,t] = fixed_effects_model(X,Y,lambda,df); 
elseif model.type == "random"
    error('Mixed/Random effects models have not been implemented yet.');
    %[beta,t] = mixed_effects_model(X,Y,lambda,df);
end

%% Compute resels
if exist('surf','var')
    surf = convert_surface(surf,'format','matlab');
    surf.faces = sort(surf.faces,2);
    edges = unique([surf.faces(:,1:2);surf.faces(:,[1,3]);surf.faces(:,2:3)],'rows');

    resels = zeros(size(edges,1),size(X,3)); 

    for ii = 1:size(Y,3)
        k = ii*(ii+1)/2; 
        normr = sqrt(slm.SSE(k,:));  % Why is this selection so "random"? Probably something in the multivariate models thats different? 
        s = 0; 
        for jj = 1:size(Y,1) % This for-loop could probably be vectorized. 
            u = Y(jj,:,ii) ./ normr;
            s = s + (u(edges(:,1)) - u(edges(:,2))).^2;
        end
        resels(:,ii) = s; 
    end
end

%% Assign variables to the output model. 
lm = struct('X',X,'beta',beta,'deg_freedom',df,'t',t);
end

function [beta,t] = fixed_effects_model(X,Y,lambda,df)
%See also SPM documentation chapter 4
% https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch8.pdf.

% Simple fixed effects linear model.
beta = (X.' * X)^-1 * X.' * Y;
SSE = sum((Y - X * beta).^2);
Vc = sum((lambda' * pinv(X).^2),2); % what does Vc stand for? I simply copied this variable name from SurfStat

contrast_estimate = lambda' * beta;
contrast_var = sqrt(Vc * SSE / df);
t = contrast_estimate ./ contrast_var + (contrast_var <=0).* (contrast_var>0);
end

function mixed_effects_model(X,Y,lambda,df)
%pass
end

function multivariate_model(X,Y,lambda,df)
%pass
end