function create_local_dataset()

% Data URLs - see http://human.brain-map.org/static/download
urls = ["http://human.brain-map.org/api/v2/well_known_file_download/178238387", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238373", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238359", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238316", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238266", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178236545"];

% Subject names
names = ["H0351.2001", "H0351.2002", "H0351.1009", "H0351.1012", "H0351.1015", "H0351.1016"];

% Make dataset directory
genetics_path = string(fileparts(fileparts(mfilename('fullpath'))));
dataset_path = genetics_path + filesep + "allen_human_brain_atlas";
mkdir(dataset_path)

disp('Downloading Allen Human Brain Atlas dataset. This may take several minutes.');
disp("Files will be downloaded to '" + dataset_path + "'.");
disp('Please be aware that this dataset takes about 4GB of space.')

% Download and upzip files.
for ii = 1:numel(urls)
    zip_file = dataset_path + filesep + "download_" + ii + ".zip";
    subject_path = dataset_path + filesep + names(ii);
    
    disp("Downloading and unzipping file " + ii + " out of " + numel(urls) + ".");
    websave(zip_file, urls{ii});
    mkdir(subject_path);
    unzip(zip_file, subject_path);
    delete(zip_file);
end
end

    