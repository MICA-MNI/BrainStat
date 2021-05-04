function [image_data, demographic_data] = fetch_tutorial_data(options)
% FETCH_TUTORIAL_DATA    Fetches data for the BrainStat tutorials.
%   image_data = FETCH_TUTORIAL_DATA('Name', Value) fetches cortical
%   thickness data on the fsaverage5 template. 
%
%   [image_data, demographic_data] = FETCH_TUTORIAL_DATA('Name', Value)
%   also returns demographic data. 
%   
%   Valid name-value pairs are:
%       n_subjects: 
%           Number of subjects. Set to 0 for all subjects, defaults to 0.
%       data_dir:  
%           Directory to store the data. Defaults to the user's home
%           directory.
%       overwrite:
%           If true, overwrites data even if it already exists. Defaults to
%           false. 

arguments 
    options.n_subjects (1,1) {mustBeInteger, mustBeNonNegative} = 0
    options.data_dir (1,1) string {mustBeFolder} = get_home_dir()
    options.overwrite (1,1) logical = false
end
options.data_dir = options.data_dir + filesep + 'brainstat_tutorial';

demographic_file = download_demographic_data(options); 
demographic_data = load_demographic_data(demographic_file, n_subjects);

image_files = download_image_data(demographic_data, options);
image_data = load_image_data(image_files); 
end


function home_dir = get_home_dir()
% GET_HOME_DIR    gets the user's home directory.
home_dir = char(java.lang.System.getProperty('user.home'));
end


function filename = download_demographic_data(options)
% Downloads the demographic data. 
if ~exist(options.data_dir, 'dir')
    mkdir(options.data_dir);
end
filename = options.data_dir + filesep + 'brainstat_tutorial_df.csv';
if ~exist(filename, 'file') || options.overwrite
    url = 'https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV/download?path=%2FSurfStat_tutorial_data&files=myStudy.csv';
    websave(filename, url);
end
end


function csv = load_demographic_data(demographic_file, n_subjects)
% Loads the demographic data.
csv = readtable(demographic_file);
if n_subjects > size(csv,1)
    error(['Requesting ', n_subjects ' subjects but only ' size(csv,1) ' are available.']);
elseif n_subjects == 0
    n_subjects = size(csv, 1);
end
csv = csv(1:n_subjects, :);
end


function filenames = download_image_data(demographic_data, options)
% Downloads image data. 
[all_files_exist, filenames] = find_image_files(demographic_data);
if ~all_files_exist
    url = 'https://box.bic.mni.mcgill.ca/s/wMPF2vj7EoYWELV/download?path=%2F&files=brainstat_tutorial.zip';
    filenames = unzip(url, options.data_dir);
end
end


function [files_exist, filenames] = find_image_files(demographic_data, options)
% Finds image files and returns whether all files were found.
subject_ids = string(demographic_data.ID2);

base_dir = string(options.data_dir) + filesep + "thickness" + filesep;
files = dir(base_dir);
files_cell = {files(:).name};
files_of_subjects = contains(files_cell, subject_ids);
filenames = base_dir + sort(files_cell(files_of_subjects));
filenames = reshape(filenames, 2, []);

expected_files = base_dir + ...
    sort(subject_ids' + ["_lh"; "_rh"] + "2fsaverage5_20.mgh");

files_exist = all(filenames(:) == expected_files(:));
end

function image_data = load_image_data(filenames)
% Loads image data.
image_data = cellfun(@(x)read_surface_data(x)', filenames, 'uniform', false);
image_data = cell2mat(image_data);
end



