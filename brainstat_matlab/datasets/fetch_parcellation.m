function parcellation = fetch_parcellation(atlas, template, n_regions, options)
% FETCH_PARCELLATION    loads a fsaverage or conte69 surface template
%   parcellation = FETCH_PARCELLATION(atlas, template, n_regions, varargin)
%   downloads and loads the 'schaefer' or 'cammoun' atlas on 'fsaverage5',
%   'fsaverage6', 'fsaverage', or 'fslr32k' (a.k.a. conte69) surface
%   template. By default, a pial surface is loaded for fsaverage templates
%   and a midthickness surface is loaded for conte69.
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

function parcellation = read_parcellation_from_targz(filename, template, parcellation, n_regions, seven_networks)
% Reads the requested file from a .tar.gz file. 
data_dir = fileparts(filename);

gunzip(filename)
untar(filename(1:end-3), data_dir)

if template == "fslr32k"
    extension = ".dlabel.nii";
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
            "_hemi-" + ["L", "R"] + "_deterministic.annot";
    otherwise
        error('Unknown parcellation %s.', parcellation);
end

if endsWith(files{1}, '.dlabel.nii')
    cii = brainstat_cifti_matlab.cifti_read(char(files));
    parcellation = cii.cdata;
elseif endsWith(files{1}, '.annot')
    parcellation = annot2parcellation(files);
    if startsWith(parcellation, 'schaefer')
        parcellation(end/2+1:end) = parcellation(end/2+1:end) + n_regions / 2;
    end
else
    error('Unknown file extension for %s.', files{1})
end
   
end

function parcellation = annot2parcellation(files)

for ii = 1:numel(files)
    [~, labels_tmp, color_table] = io_utils.freesurfer.read_annotation(files{ii});
    [vertex_id, labels_compress] = find(labels_tmp == color_table.table(:,5)');
    [~, indices] = sort(vertex_id);
    labels{ii} = labels_compress(indices);

    % Sanity check that we find the correct number of labels:
    if numel(labels{ii}) ~= numel(labels_tmp)
        error('Woops! Seems like something is wrong with this .annot file.');
    end
end
parcellation = [labels{1}; labels{2}];
end

