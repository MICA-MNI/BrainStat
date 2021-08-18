function mpc = compute_mpc(profile, labels)
% COMPUTE_MPC    computes MPC from histology profiles
%   mpc = COMPUTE_MPC(profile, labels, template) computes microstructural
%   profile covariance (mpc) from the provided histological profiles.
%   Labels is a vector containing the region labels of each vertex.
%
%   See also COMPUTE_HISTOLOGY_GRADIENTS, READ_HISTOLOGY_PROFILE.

arguments
    profile (:,:) {mustBeFloat}
    labels {mustBeVector}
end

roi_profile = labelmean(profile', labels(:)');
partial_r = partialcorr(roi_profile, mean(profile, 1)');
mpc = 0.5 * log((1 + partial_r) ./ (1 - partial_r)); 
mpc(isnan(mpc)) = 0;
mpc(isinf(mpc)) = 0;
end