function stats = brainstat_pls(x,y,varargin)
% BRAINSTAT_PLS   Computes a partial least squares regression and statistics.
%   stats = BRAINSTAT_PLS(x,y) computes a partial least squares regression of
%   predictor variable x and response variable y. Returns a structure stats
%   containing: 
%       x: normalized x values 
%       y: normalized y values
%       U: rotation for x 
%       V: rotation for y
%       SV_p: p-value of each component
%       U_confidenceinterval: confidence interval of U
%       V_confidenceinterval: confidence interval of V
%       R_real: correlation between rotated x and y
%       R_kfold: correlations between x and y from k-fold cross-validation.
%
%   stats = BRAINSTAT_PLS(x,y,'Name','Value') allows for modifying several
%   parameters. Valid names are: 
%       'rng'
%           initiation of RNG, see help rng(); default 'shuffle'
%       'n_perm'
%           number of permutations; default 1000
%       'n_boot'
%           number of bootstraps; default 1000
%       'K'
%           number of K-folds; default 10
%       'CI_alpha' 
%           alpha for confidence intervals; default 0.95.
%       'n_workers'
%           the size of the parallel pool to use. 
%       'delete_pool'
%           Logical, if true delete the parallel pool at the end of 
%           the function. Default: true if no parallel pool opened 
%           before the function is called, false otherwise.
%
%   TODO: ADD BRAINSTAT READTHEDOCS HYPERLINK.

%% Input handling.
% Deal with varargin.
p = inputParser;
is_scalar_numeric = @(v)numel(v) == 1 && isnumeric(v);
is_alpha = @(v) is_scalar_numeric(v) && v > 0 && v < 1;
addParameter(p,'n_perm',1000,is_scalar_numeric);
addParameter(p,'n_boot',1000,is_scalar_numeric);
addParameter(p,'rng','shuffle');
addParameter(p,'K',10);
addParameter(p,'CI_alpha',0.95,is_alpha);
addParameter(p,'n_workers',0,is_scalar_numeric);
addParameter(p,'delete_pool',nan,@islogical);

parse(p, varargin{:});
I = p.Results; 

% Set RNG
rng(I.rng);

% Make sure the input is of the correct type.
x = double(x);
y = double(y);

% Normalize input. 
x = zscore(x);
y = zscore(y);

% Get true loadings and scores. Two steps to optimize speed (see local_pls function). 
[U,~,V] = local_pls(x,y,true);
[~,S] = local_pls(x,y);

% % If pool deletion is not specified then delete the pool only if none is
% % open yet. 
if license('test','Parallel_Computing_Toolbox')
    if isnan(I.delete_pool)
        I.delete_pool = isempty(gcp('nocreate'));
    end

    % Open a parallel pool
    if I.n_workers > 0 
        local_modifypool(I.n_workers);
    end
elseif I.n_workers ~= 0
    warning(['Parallel computing was requested, but the Parallel Computing'  ...
        ' Toolbox was not found. Continuing without parallel processing.']);
    I.delete_pool = false; % Ascertain the function doesn't try to delete the pool at the end. 
    I.n_workers = 0; 
end

%% Permutation test.
if I.n_perm ~= 0
    disp('Running permutation test.');
    S_randi = zeros(size(x,2),I.n_perm);
    parfor (ii = 1:I.n_perm,I.n_workers)
        [~,S_randi(:,ii)] = local_pls(x(randperm(size(x,1)),:), y); %#ok<PFBNS>
    end
    SV_p = mean(S < S_randi, 2); % P-value for permutation test.  
else
    SV_p = [];
end

%% Bootstrap.
if I.n_boot ~= 0
    disp('Running bootstrap.');
    % Convert CI Alpha to Z value.
    z = -sqrt(2) * erfcinv(I.CI_alpha);

    % Bootstrap x and y and calculate their U's and V's.    
    if license('test','statistics_toolbox')
        opt = statset('UseParallel', I.n_workers~=0);
        bootstat = bootstrp(I.n_boot, @local_pls, x, y, true, 'Options', opt);
    else
        warning(['Could not find the Statistics and Machine Learning Toolbox. ' ...
            'We''ll use our bootstrap implementation instead of Mathworks'' implementation. ' ...
            'Be aware that this implementation is more than an order of magnitude slower.'])
        bootstat = local_bootstrap(@local_pls, I.n_boot, I.n_workers, ...
            numel(U) + numel(V), x, y, true);
    end
    
    % Extract bootstrapped U and V from output array.
    U_boot = bootstat(:, 1:numel(U)); 
    V_boot = bootstat(:, numel(U)+1:end); 
    
    % Compute confidence intervals i.e. CI_x = x_bar +- (SEM * z)
    CI_U = U' + [std(U_boot); -std(U_boot)] / sqrt(I.n_boot) * z; 
    CI_V = V' + [std(V_boot); -std(V_boot)] / sqrt(I.n_boot) * z; 
else
    CI_U = []; CI_V = []; 
end
%% K-fold
% Compute real correlation. 
R_real = corr(x*U,y*V);

% Grab indices.
if I.K ~= 0
    disp('Running k-fold crossvalidation.');
    % Create indices for K groups of as close to equal size as possible.  
    groups = repelem(1:ceil(size(x,1) / I.K), I.K);
    cv_idx = groups(randperm(numel(groups), size(x,1)));

    % Train on all but one set, test on the holdout.
    R_kfold = zeros(I.K, 1);
    parfor (ii = 1:I.K, I.n_workers)
        train = cv_idx~=ii; 
        [U_train,~,V_train] = local_pls(x(train,:), y(train,:),true);  %#ok<PFBNS>
        R_kfold(ii) = corr(x(~train,:) * U_train, y(~train,:) * V_train); 
    end
else
    R_kfold = []; 
end

%% Save data and close parallel pool. 
stats = struct('U', U, ...
               'V', V, ...
               'x', x, ...
               'y', y, ...
               'U_confidenceinterval', CI_U, ...
               'V_confidenceinterval', CI_V, ...
               'R_real', R_real, ...
               'R_kfold',R_kfold, ...
               'SV_p', SV_p); 
        
if I.delete_pool
    delete(gcp('nocreate'));
end
end
%% Local functions
function varargout = local_pls(x,y,econ)
% Run a PLS.
if nargin < 3
    econ = false;
end
R = corr(x,y);

% Several options to speed up the computations.
% 1 - If two output arguments are requrested, assume the user only cares
% about S. 
% 2 - If econ is true, compute only the first vectors.
% 3 - If neither of the above applies perform the full SVD.
if nargout == 2
    % Assume only S is of interest.
    [~,S] = svd(R);
elseif econ
    % If econ is true, compute only the first vector.
    [U,S,V] = svds(R,1);
else
    % If neither of the above, perform the full SVD. THIS IS SLOW!
    [U,S,V] = svd(R);
end

% If one output argument requested (MATLAB's bootstrp function), then
% concatenate U and V. If two output arguments are requested, return only
% S. If three are requested, return the first vectors only.
if nargout == 1
    varargout{1} = [U(:,1);V(:,1)];
elseif nargout == 2
    varargout{1} = [];
    varargout{2} = diag(S); 
else
    varargout{1} = U(:,1);
    varargout{2} = diag(S);
    varargout{3} = V(:,1);
end
end

function p = local_modifypool(n_workers)
% Modifies the parallel pool to have the selected number of workers. 
p = gcp('nocreate');
if ~isempty(p)
    if p.NumWorkers == n_workers
        return
    else
       delete(p)
    end
end
p = parpool(n_workers);
end

function val = local_bootstrap(fun,n_boot,n_workers,size_out,varargin)
% Brainstat implementation of MATLAB's bootstrp() function. Note that using
% the MATLAB implementation is preferred as its much faster than this one. 

% Find number of rows in non-scalar input. 
sizes = cellfun(@(x)size(x,1),varargin);
sizes(sizes==1) = []; 
if isempty(sizes)
    error('Cannot bootstrap with only scalar input or no input.');
elseif numel(unique(sizes)) > 1
    error('Non-scalar input to bootstrap must have the same number of rows.')
end
sizes = sizes(1);

% Initialize val array. 
val = zeros(n_boot,size_out); 

parfor (ii = 1:n_boot, n_workers)
    % Indices of resampling.
    R = randi(sizes,sizes);
    
    % Resample.
    boot_arg = cell(size(varargin));
    for jj = 1:numel(varargin)
        if ~isscalar(varargin{jj})
            boot_arg{jj} = varargin{jj}(R,:);
        else
            boot_arg{jj} = varargin{jj};
        end
    end
    
    % Run the bootstrap function. 
    val(ii,:) = fun(boot_arg{:});  %#ok<PFBNS>
end
end