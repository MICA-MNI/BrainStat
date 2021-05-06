function surfaces = fetch_surfaces(options)
% FETCH_SURFACES    Fetches tutorial surface data.
%   surfaces = fetch_surfaces('Name', Value) fetches surfaces of the
%   fsaverage5 template. 

%   Valid name-value pairs are: 
%       template:
%           Must be 'fsaverage5' (more to be added), defaults to
%           'fsaverage5'
%       data_dir:  
%           Directory to store the data. Defaults to the user's home
%           directory.
%       join:
%           If true, returns combined left/right surfaces, otherwise
%           returns a cell array where the first surface is left, and the
%           second is right. Defaults to true. 

arguments
    options.template (1,:) char = 'fsaverage5'
    options.data_dir (1,:) string {mustBeFolder} = brainstat_utils.get_home_dir();
    options.join (1,1) logical = true
end

surface_files = get_surface_files(options);
surfaces = cellfun(@read_surface, surface_files, 'uniform', false);

if options.join
    surfaces = combine_surfaces(surfaces{:});
end

end

function surface_files = get_surface_files(options)
% Downloads surfaces and gets the filenames.

surface_dir = options.data_dir + filesep + 'brainstat_surfaces';
surface_files = surface_dir + filesep + options.template + "_" + ["lh", "rh"] + ".surf.gii";

% Download if necessary. 
if ~all(cellfun(@(x) exist(x, 'file'), surface_files))
    url = get_surface_urls(options.template);
    unzip(url, surface_dir);
end
end

function url = get_surface_urls(template)
% Fetches the URL of a surface. 

urls = struct( ...
    'fsaverage5', 'https://box.bic.mni.mcgill.ca/s/6J2xLne6Tmw07uR/download' ...
);

try
    url = urls.(template);
catch err
    if err.identifier == "MATLAB:nonExistentField"
        error('Invalid surface template.');
    else
        throw(err);
    end
end
end

