function [thickness, demographics] = fetch_mics_data(data_dir)
% FETCH_MICS_DATA    loads the MICS tutorial data.
%
% [thickness, demographics] = FETCH_MICS_DATA() downloads the MICs data to
% the brainstat data directory and returns the cortical thickness and
% demographic data.
%
% ... = FETCH_MICS_DATA(data_dir) specifies the directory to download the
% data to.

arguments
    data_dir (1,1) string = brainstat_utils.get_brainstat_directories('mics_data_dir')
end

thickness_file = data_dir + filesep + "mics_tutorial_thickness.h5";
demographics_file = data_dir + filesep + "mics_tutorial_participants.csv";
json = brainstat_utils.read_data_fetcher_json();

if ~exist(thickness_file, 'file')
    websave(thickness_file, json.mics_tutorial.thickness.url);
end
if ~exist(demographics_file, 'file')
    websave(demographics_file, json.mics_tutorial.participants.url);
end

demographics = readtable(demographics_file);
demographics = demographics(:, 2:end);
thickness = h5read(thickness_file, '/thickness')';
end