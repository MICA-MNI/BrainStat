function gm = compute_histology_gradients(mpc, options)
% COMPUTE_HISTOLOGY_GRADIENTS   computes gradients from MPC
%   gm = COMPUTE_HISTOLOGY_GRADIENTS(mpc, varargin) computes gradients from
%   microstructural profile covariance. Optional arguments contain all
%   name-value arguments for GradientMaps and its fit functions.
%   For full details please consult the help of GradientMaps. Allowed names
%   are: 'kernel', 'approach', 'n_components', 'alignment', 'random_state',
%   'gamma', 'sparsity', 'reference', 'n_iter'.
%
%   See also GRADIENTMAPS, COMPUTE_MPC.
arguments 
    mpc (:,:)
    options.kernel (1,:) char = 'na'
    options.approach (1,:) = 'dm'
    options.n_components (1,1) = 10
    options.alignment (1,:) char = 'none'
    options.random_state (1,1) = nan
    options.gamma (1,1) {mustBePositive} = 1 / size(mpc,1)
    options.sparsity (1,1) {mustBeNonnegative} = 0.9
    options.reference = nan
    options.n_iter (1,1) {mustBeInteger, mustBePositive} = 10
end

gm = GradientMaps('kernel', options.kernel, ...
                  'approach', options.approach, ...
                  'n_components', options.n_components, ...
                  'alignment', options.alignment, ...
                  'random_state', options.random_state);
gm = gm.fit(mpc, 'gamma', options.gamma, 'sparsity', options.sparsity, 'niterations', ...
    options.n_iter, 'reference', options.reference);

end

