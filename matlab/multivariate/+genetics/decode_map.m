function [t,p] = decode_map(nii_file,varargin)

% Grab input arguments.
p = inputParser;
addParameter(p, 'gene_name', 'SLC6A2');
addParameter(p, 'inference_method', 'random');
addParameter(p, 'probes_reduction_method', 'average');
addParameter(p, 'mask', nan);
addParameter(p, 'radius', 4);
addParameter(p, 'probe_exclusion_keyword',nan);
addParameter(p, 'flip_hemisphere',true);

parse(p, varargin{:})
I = p.Results; 

% Get the probes and expression associated with the genes. 
[probe_id, probe_name] = genetics.api.get_probes_from_genes(I.gene_name);
[expression, well_ids, donor_names] = genetics.api.get_expression_from_probes(probe_id);

% Combine expression values.
if I.probes_reduction_method == "average"
    expression = mean(expression,2);
elseif method == "pca"
    [~,comp] = pca(expression);
    expression = comp(:,1);
end

% Get the mean value for every point in the volume within a sphere. 
[~,hdr] = read_volume(nii_file);
voxel_size = hdr.dime.pixdim(2:4);
if ~all(voxel_size == voxel_size(2))
    % Currently the computation of the sphere cannot be performed for
    % non-isotropic vertices. This check can be removed once that's fixed. 
    error('Can only handle isotropic data currently.')
end

% Get the mni matrix coordinates of the wells.
mni_coordinates = genetics.utilities.get_coordinates_from_well_ids(well_ids,voxel_size(1));

% Discard probes in the mask. 
nifti_values = genetics.utilities.get_values_at_locations(nii_file, mni_coordinates, I.radius, I.mask);

% Perform statisticla test
if I.inference_method == "random"
    [t,p] = genetics.effects.random(nifti_values, expression, donor_names);
else
    error('Inference method not implemented.');
end
end