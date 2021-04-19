function dataset_path = create_genetics_dataset(dataset_path)
% CREATE_GENETICS_DATASET    Downloads the AHBA dataset.
%
%   dataset_path = create_genetics_dataset(dataset_path) downloads the AHBA
%   dataset to the designated path. If no path is provided, the dataset is
%   downloaded to the BrainStat toolbox. Returns the dataset path as a
%   string scalar.


if ~exist('dataset_path', 'var')
    dataset_path = [];
end

% Get dataset directory
if isempty(dataset_path)
    genetics_path = string(fileparts(fileparts(mfilename('fullpath'))));
    dataset_path = genetics_path + filesep + "allen_human_brain_atlas";
else
    dataset_path = string(dataset_path);
end
microarray_path = dataset_path + filesep + "microarray";

% Subject names
%names = ["H0351.2001", "H0351.2002", "H0351.1009", "H0351.1012", "H0351.1015", "H0351.1016"];
names = ["9861", "10021", "12876", "14380", "15496", "15697"];

% Data URLs - see http://human.brain-map.org/static/download
urls = ["http://human.brain-map.org/api/v2/well_known_file_download/178238387", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238373", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238359", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238316", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178238266", ...
        "http://human.brain-map.org/api/v2/well_known_file_download/178236545"];

% Check if all files exist
subject_paths = microarray_path + filesep + "normalized_microarray_donor" + names;
files_exist = all(cellfun(@subject_files_exist, subject_paths));
if ~files_exist
    disp('We could not find the AHBA dataset on your computer.');
    disp('May we download it (size: ~4GB)?')
    disp(['The download directory was set to', char(dataset_path) '.'])
    disp('If you''d prefer a different directory, then please set the data_dir in the calling function.');
    disp('');
    answer = input('Download the AHBA dataset? (y/n)', 's'); 
    if lower(answer) ~= "y"
        error('Cannot continue without downloading the dataset.');
    end

    % Download and upzip files.
    if ~exist(dataset_path, 'dir')
        mkdir(dataset_path)
    end
    if ~exist(microarray_path, 'dir')
        mkdir(microarray_path)
    end
    for ii = 1:numel(urls)
        zip_file = dataset_path + string(filesep) + "download_" + ii + ".zip";

        if ~subject_files_exist(subject_paths(ii))
            mkdir(subject_paths(ii));
            disp("Downloading and unzipping file " + ii + " out of " + numel(urls) + ".");
            websave(zip_file, urls{ii});
            unzip(zip_file, subject_paths(ii));
            delete(zip_file);
        end
    end
    disp('Finished downloading the AHBA dataset.');
end
end

function files_exist = subject_files_exist(subject_path)
output_files = {'MicroarrayExpression.csv',	'Probes.csv', 'Ontology.csv', ...
        'Readme.txt', 'PACall.csv', 'SampleAnnot.csv'};
files_exist = all(cellfun(@(x)exist(string(subject_path) + filesep + x, 'file'), ...
            output_files));
end
    