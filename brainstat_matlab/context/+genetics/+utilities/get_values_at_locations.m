function [values,xyz_coordinates] = get_values_at_locations(nii_file, location, radius, mask_file)

% Load nifti file
nii = load_nii(nii_file);

% To match python implementation - DOUBLE CHECK CORRECTNESS
nii.img = nii.img(end:-1:1,:,:); 

% Convert location to XYZ coordinates
xyz_coordinates = round(genetics.utilities.world_to_and_from_voxel(location, nii_file, 'xyz'));

% Set the mask.
if ~isnan(mask_file)
    mask = read_volume(mask_file);
else
    mask = ~isnan(nii.img) & nii.img~=0;
end

% Compute average within ROIs. 
values = zeros(size(xyz_coordinates,1),1);
for ii = 1:size(xyz_coordinates,1)
    coord_data = xyz_coordinates(ii,:);
    if ~isnan(radius)
        sphere_mask = get_sphere_mask(nii.img, nii.hdr, xyz_coordinates(ii,:), radius);
    else
        sphere_mask = zeros(size(mask),'logical');
        sphere_mask(coord_data(1),coord_data(2),coord_data(3)) = true; 
    end
    roi = sphere_mask & mask;
    values(ii) = mean(nii.img(roi));
end
end

function sphere = get_sphere_mask(vol,header,target,radius)
% Produces a sphere of size radius centered on the target location within
% the volume. Currently it can only handle isotropic voxels. 

vs = header.dime.pixdim(2:4);
if ~all(vs == vs(1))
    error('Non-isotropic voxel sizes are not currently supported.');
end
sz = size(vol);


[X,Y,Z] = ndgrid(1:sz(1), 1:sz(2), 1:sz(3));
X = X - target(1); Y = Y - target(2); Z = Z - target(3); 
R = sqrt(X.^2 + Y.^2 + Z.^2);

sphere = R <= (radius./vs(1));
end