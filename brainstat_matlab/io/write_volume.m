function write_volume(filename,vol,varargin)

p = inputParser;
addParameter(p,'voxelsize',[]);
addParameter(p,'origin',[]);
addParameter(p,'datatype',[])
addParameter(p,'description',[])
parse(p,varargin{:}); 

if endsWith(filename,{'.nii','.nii.gz'})
    nii = io_utils.nifti_toolbox.make_nii(vol, p.Results.voxelsize, p.Results.origin, ...
        p.Results.datatype,p.Results.description);
    io_utils.nifti_toolbox.save_nii(nii,filename)
else
    error('Only the writing of nifti files is supported.');
end
end
