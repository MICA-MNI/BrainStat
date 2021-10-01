function [volume,header] = read_volume(file)
% READ_VOLUME    Reads a volume file. 
% [volume, header] = READ_VOLUME(file) reads the nifti/minc volume in the
% input file. Returns the image data in "volume" and header data in
% "header".
%
% ADD LINK TO BRAINSTAT DOCUMENTATION. 

arguments
    file (1,:) char
end

% Niftiread is part of the image processing toolbox 
if endsWith(file,{'.nii.gz','.nii'})
    nii = io_utils.nifti_toolbox.load_nii(file);
    volume = nii.img; 
    header = nii.hdr;
else
    error('Unrecognized volume file. Currently supported volumes are .nii.gz, .nii.');
end
end