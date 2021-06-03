function mpc = compute_mpc(profile, labels, template)
% COMPUTE_MPC    computes MPC from histology profiles
%   mpc = COMPUTE_MPC(profile, labels, template) computes microstructural
%   profile covariance (mpc) from the provided histological profiles.
%   Labels is a vector containing the region labels of each vertex.
%   If template is provided, then a correction for y-coordinate is
%   performed. Template may be 'fs_LR_64k'.
%
%   See also COMPUTE_HISTOLOGY_GRADIENTS, READ_HISTOLOGY_PROFILE.

arguments
    profile (:,:) {mustBeFloat}
    labels {mustBeVector}
    template (1,1) string = ""
end

if ~isempty(template{1})
    profile = y_correction(profile, template);
end

roi_profile = labelmean(profile', labels(:)');
partial_r = partialcorr(roi_profile, mean(roi_profile, 2));
mpc = 0.5 * log((1 + partial_r) ./ (1 - partial_r)); 
mpc(isnan(mpc)) = 0;
mpc(isinf(mpc)) = 0;
end

function residuals = y_correction(profile, template)
    surface = template_to_surface(template);
    predictor = [ones(size(surface.coord,2), 1), surface.coord(2,:)'];
    linear_regression = @(X,Y) Y - X * ((X'*X)^-1 * X' * Y);
    residuals = linear_regression(predictor, profile);
end

function surface = template_to_surface(template)

switch template
    case 'fs_LR_64k'
        [left_surface, right_surface] = load_conte69();
        surface = combine_surfaces(left_surface, right_surface);
    otherwise
        error('Currently only ''fs_LR_64k'' is accepted as template.');
end
end