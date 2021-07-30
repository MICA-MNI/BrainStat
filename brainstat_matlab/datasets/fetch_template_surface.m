function [surf_lh, surf_rh] = fetch_template_surface(template, options)
% FETCH_TEMPLATE_SURFACE    loads a fsaverage or conte69 surface template
%   [surf_lh, surf_rh] = FETCH_TEMPLATE_SURFACE(template, varargin)
%   downloads and loads a 'fsaverage3', 'fsaverage4', 'fsaverage5',
%   'fsaverage6', 'fsaverage', or 'conte69' (a.k.a. 'fslr32k') surface
%   template. By default, a pial surface is loaded for fsaverage templates
%   and a midthickness surface is loaded for conte69.
%
%   Valid name-value pairs are:
%       'data_dir': a char containing the path to the location to store the
%           data. Defaults to ${HOME_DIRECTORY}/brainstat_data/surfaces.
%       'layer': the specific surface to load. Valid values for fsaverage
%           are 'pial' (default), 'white', 'sphere', 'smoothwm',
%           'inflated'. Valid values for conte69 are: 'midthickness'
%           (default), 'inflated', 'vinflated'.


arguments
    template (1,:) char
    options.data_dir (1,:) string {mustBeFolder} = brainstat_utils.get_data_dir( ...
        'subdirectory', 'surfaces', 'mkdir', true);
    options.layer char = ''
end

switch lower(template)
    case {'fslr32k', 'conte69'}
        if isempty(options.layer)
            options.layer = 'midthickness';
        end
    otherwise
        if isempty(options.layer)
            options.layer = 'pial';
        end
end
filename = dataset_utils.download_OSF_files(template, 'data_dir', options.data_dir);
[surf_lh, surf_rh] = read_surface_from_targz(filename, template, options.layer);

end


function [surf_lh, surf_rh] = read_surface_from_targz(filename, template, layer)
% Reads the requested file from a .tar.gz file. 
data_dir = fileparts(filename);

gunzip(filename)
untar(filename(1:end-3), data_dir)

switch template
    case {'fslr32k', 'conte69'}
        files = string(data_dir) + filesep + "tpl-conte69" + filesep + ...
            "tpl-conte69_space-MNI305_variant-fsLR32k_" + layer + "." + ["L", "R"] + ...
            ".surf.gii";
    otherwise
        files = string(data_dir) + filesep + "tpl-fsaverage" + filesep + template + ...
            filesep + "surf" + filesep + ["lh.", "rh."] + layer; 
end
surf_lh = read_surface(files{1});
surf_rh = read_surface(files{2});

end