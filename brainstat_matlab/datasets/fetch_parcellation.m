function parcellation = fetch_parcellation(template, atlas, n_regions, options)
% FETCH_PARCELLATION    loads a fsaverage or conte69 surface template
%   parcellation = FETCH_PARCELLATION(template, atlas, n_regions, varargin)
%   downloads and loads the 'cammoun', 'glasser', 'schaefer', or 'yeo',
%   atlas on 'fsaverage5', 'fsaverage6', 'fsaverage', or 'fslr32k' (a.k.a.
%   conte69) surface template. 
%
%   Supported number of regions for each atlas are as follows:
%       Cammoun: 33, 60, 125, 250, 500. 
%       Glasser: 360 (Note fsaverage6 not supported).
%       Schaefer: 100, 200, 300, 400, 500, 600, 800, 1000. 
%       Yeo: 7, 17
%
%   Valid name-value pairs are:
%       'data_dir': a char containing the path to the location to store the
%           data. Defaults to ${HOME_DIRECTORY}/brainstat_data/surfaces.
%       'seven_networks': If true, uses the 7 network schaefer parcellation,
%           otherwise uses the 17 network verison. Defaults to true. 


arguments
    template (1,:) char
    atlas (1,:) char
    n_regions (1,1) double
    options.data_dir (1,:) string = brainstat_utils.get_brainstat_directories('parcellation_data_dir')
    options.seven_networks (1,1) logical = true
end

atlas = lower(atlas);

if atlas == "conte69"
    atlas = 'fslr32k';
end

switch lower(atlas)
    case {'schaefer', 'cammoun'}
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
    case 'glasser'
        parcellation = fetch_glasser_parcellation(template, options.data_dir);
    case "yeo"
        parcellation = fetch_yeo_parcellation(template, n_regions, options.data_dir);
    otherwise
        error("Unknown atlas: '%s'.", atlas)
end
end

function labels = read_parcellation_from_targz(filename, template, atlas, n_regions, seven_networks)
% Reads the requested file from a .tar.gz file. 

% Get filenames
temp_dir = tempname;
[~, tar_name] = fileparts(filename);
tar_file = temp_dir + string(filesep) + tar_name;
[target_files, all_files, sub_dir] = get_label_file_names(temp_dir, template, atlas, n_regions, seven_networks);

% Setup a cleanup step. 
mkdir(temp_dir);
cleaner = onCleanup(@()clean_temp_dir(temp_dir, all_files, tar_file, sub_dir, template));

% Read data
gunzip(filename, temp_dir)
untar(tar_file, temp_dir)
labels = read_surface_data(target_files);

% Bring to output format.
if iscell(labels)
    labels = cell2mat(labels(:));
end

if endsWith(target_files{1}, '.annot')
    labels = labels - 1;
    if startsWith(atlas, 'schaefer')
        labels(end/2+1:end) = (labels(end/2+1:end) + n_regions / 2) .* (labels(end/2+1:end)>0);
    end
end
end

function [target_files, all_files, sub_dir] = get_label_file_names(temp_dir, template, parcellation, n_regions, seven_networks)
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

if parcellation == "schaefer"
    all_roi = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
elseif parcellation == "cammoun"
    all_roi = [33, 60, 125, 250, 500];
end

switch parcellation
    case {'schaefer', 'schaefer2018'}    
        if extension == ".dlabel.nii"
            hemi = "_hemi-LR_desc-";
        else
            hemi = "_hemi-" + ["L"; "R"] + "_desc-";
        end
        sub_dir = "atl-schaefer2018";
        file_dir = string(temp_dir) + filesep + sub_dir + filesep + template;
        target_files = file_dir + filesep + "atl-Schaefer2018_space-" + template + ...
            hemi + n_regions + "Parcels" + n_networks + ...
            "Networks_deterministic" + extension;
        all_files = file_dir + filesep + "atl-Schaefer2018_space-" + template + ...
            hemi + all_roi + "Parcels" + cat(3, 7, 17) + ...
            "Networks_deterministic" + extension;
    case {'cammoun', 'cammoun2012'}
        sub_dir = "atl-cammoun2012";
        file_dir = string(temp_dir) + filesep + sub_dir + filesep + template;
        target_files = file_dir +  filesep + "atl-Cammoun2012_space-" + template + "_res-" + ...
            sprintf('%03d', n_regions) + "_hemi-" + ["L", "R"] + "_deterministic" + extension;
        all_files = file_dir +  filesep + "atl-Cammoun2012_space-" + template + "_res-" + ...
            cellfun(@(x)sprintf("%0.3d", x), num2cell(all_roi)) + "_hemi-" + ["L"; "R"] + ...
            "_deterministic" + extension;
    otherwise
        error('Unknown parcellation %s.', parcellation);
end


end

function clean_temp_dir(temp_dir, all_files, tar_file, sub_dir, template)
cellfun(@delete, all_files);
delete(tar_file)
rmdir(string(temp_dir) + filesep + sub_dir + filesep + template)
rmdir(string(temp_dir) + filesep + sub_dir);
rmdir(temp_dir);
end

function parcellation = fetch_glasser_parcellation(template, data_dir)
    json = brainstat_utils.read_data_fetcher_json();
    urls = json.parcellations.glasser.(template).url; 
 
    filenames = data_dir + filesep + "glasser_360_" + template + "_" + ["lh", "rh"] + ".label.gii";
    if ~all(cellfun(@(x) exist(x, 'file'), filenames))
        for ii = 1:numel(filenames)
            websave(filenames{ii}, urls{ii});
        end
    end
    parcellations = read_surface_data(filenames);
    parcellation = [double(parcellations{1}); double(parcellations{2}) + 180*(parcellations{2}>0)];   
end

function parcellation = fetch_yeo_parcellation(template, n_regions, data_dir)
    json = brainstat_utils.read_data_fetcher_json();
    url = json.parcellations.yeo.url; 
    
    filenames = data_dir + filesep + template + "_" + ["lh", "rh"] + "_yeo" + num2str(n_regions) + ".label.gii";
    if ~all(cellfun(@(x) exist(x, 'file'), filenames))
        zipfile = tempname + ".zip";
        cleaner = onCleanup(@(x) delete(zipfile));
        websave(zipfile, url);
        unzip(zipfile, data_dir);
    end

    parcellations = read_surface_data(filenames);
    parcellation = [double(parcellations{1}); double(parcellations{2})];   
end

