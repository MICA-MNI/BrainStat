function gradients = fetch_gradients(template, name, options)
% FETCH_GRADIENTS    fetches example gradients.
% gradients = FETCH_GRADIENTS(template, name, varargin) fetches gradients
% on the specified template surface. Variable name denotes the name of
% gradients. Valid values for template are: "fsaverage5" (default),
% "fsaverage", "fslr32k". Valid values for name are "margulies2016". The
% following name-value pairs are accepted:
%
%   'data_dir': Directory to store the downloaded data files, defaults to 
%       $HOME_DIR/brainstat_data/gradient_data.
%   'overwrite': If true, overwrite existing data files, defaults to false.
%   
% Note: margulies2016 gradients were computed from the mean HCP-S1200
% functional connectome using BrainSpace with the following parameters:
% cosine similarity kernel, diffusion embedding, alpha=0.5, sparsity=90,
% and diffusion_time=0. Gradients were computed on the fsaverage5 surface
% and interpolated to the other surfaces. 

arguments
    template (1,1) string = "fsaverage5"
    name (1,1) string = "margulies2016"
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('gradient_data_dir')
    options.overwrite (1,1) logical = false
end

gradients_file = options.data_dir + filesep + name + "_gradients.h5";
if ~exist(gradients_file, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(gradients_file, json.gradients.(name).url);
end

gradients = h5read(gradients_file, "/" + template);