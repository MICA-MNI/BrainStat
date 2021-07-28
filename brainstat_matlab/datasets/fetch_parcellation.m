function parcellation = fetch_parcellation(atlas, template, n_regions, options)
% FETCH_PARCELLATION    loads a fsaverage or conte69 surface template
%   parcellation = FETCH_PARCELLATION(atlas, template, n_regions, varargin)
%   downloads and loads the 'schaefer' or 'cammoun' atlas on 'fsaverage5',
%   'fsaverage6', 'fsaverage', or 'fslr32k' (a.k.a. conte69) surface
%   template. By default, a pial surface is loaded for fsaverage templates
%   and a midthickness surface is loaded for conte69.
%
%   Supported number of regions for the Schaefer atlas are: 100, 200, 300,
%   400, 500, 600, 800, 1000. Supported number of regions for the Cammoun
%   atlas are: 33, 60, 125, 250, 500.
%
%   Valid name-value pairs are:
%       'data_dir': a char containing the path to the location to store the
%           data. Defaults to ${HOME_DIRECTORY}/brainstat_data/surfaces.
%       'seven_networks': If true, uses the 7 network schaefer parcellation,
%           otherwise uses the 17 network verison. Defaults to true. 


arguments
    atlas (1,:) char
    template (1,:) char
    n_regions (1,1) double
    options.data_dir (1,:) string {mustBeFolder} = brainstat_utils.get_data_dir( ...
        'subdirectory', 'surfaces', 'mkdir', true);
    options.seven_networks (1,1) logical = true
end

if template == "conte69"
    template = 'fslr32k';
end

filename = dataset_utils.download_OSF_files(...
    template, ...
    'data_dir', options.data_dir, ...
    'parcellation', atlas);

parcellation = read_parcellation_from_targz(...
    filename, ...
    template, ...
    atlas, ...
    n_regions, ...
    options.seven_networks);

end

function labels = read_parcellation_from_targz(filename, template, parcellation, n_regions, seven_networks)
% Reads the requested file from a .tar.gz file. 
data_dir = fileparts(filename);

gunzip(filename)
untar(filename(1:end-3), data_dir)

if template == "fslr32k"
    if parcellation == "schaefer"
        extension = ".dlabel.nii";
    else
        extension = ".label.gii";
    end
else
    extension = ".annot";
end

if seven_networks
    n_networks = 7;
else
    n_networks = 17;
end

switch parcellation
    case {'schaefer', 'schaefer2018'}    
        if extension == ".dlabel.nii"
            hemi = "_hemi-LR_desc-";
        else
            hemi = "_hemi-" + ["L", "R"] + "_desc-";
        end
        files = string(data_dir) + filesep + "atl-schaefer2018" + filesep + ...
            template + filesep + "atl-Schaefer2018_space-" + template + ...
            hemi + n_regions + "Parcels" + n_networks + ...
            "Networks_deterministic" + extension;
    case {'cammoun', 'cammoun2012'}
        files = string(data_dir) + filesep + "atl-cammoun2012" + filesep + template + ...
            filesep + "atl-Cammoun2012_space-" + template + "_res-" + sprintf('%03d', n_regions) + ...
            "_hemi-" + ["L", "R"] + "_deterministic" + extension;
    otherwise
        error('Unknown parcellation %s.', parcellation);
end

labels = read_surface_data(files);
if iscell(labels)
    labels = cell2mat(labels(:));
end

if endsWith(files{1}, '.annot')
    labels = labels - 1;
    if startsWith(parcellation, 'schaefer')
        labels(end/2+1:end) = labels(end/2+1:end) + n_regions / 2;
    end
end
end


