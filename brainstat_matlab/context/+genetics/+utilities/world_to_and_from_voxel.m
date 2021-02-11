function out_coord = world_to_and_from_voxel(coord, nii_file, transform_to)
% Transforms between nifti world coordinates and matrix coordinates (in
% MATLAB format i.e. start counting from 1). 

nii = load_nii(nii_file);
affine = [nii.hdr.hist.srow_x;
          nii.hdr.hist.srow_y;
          nii.hdr.hist.srow_z;
          0 0 0 1];

switch lower(transform_to)
    case 'xyz'
        out_coord = affine^-1 * [coord'; ones(1,size(coord,1))]+1; % add one because MATLAB starts counting from 1.
    case 'world'
        out_coord = affine * [coord'-1'; ones(1,size(coord,1))]; 
end
out_coord = out_coord(1:3,:)';
end