function [thickness, demographics] = fetch_mics_data(options)
% FETCH_MICS_DATA    fetches thickness and demographic MICS data
%   [thickness, demographics] = FETCH_MICS_DATA(data_dir) fetches MICS
%   cortical thickness and demographics data. The following name-value
%   pairs are allowed:
%
%   'data_dir'
%       The directory to save the data. Defaults to
%       $HOME/brainstat_data/mics_data
%   'overwrite'
%       If true, overwrites older files. Defaults to false.

arguments
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('mics_data_dir')
    options.overwrite (1,1) logical = false
end

if ~exist(options.data_dir, 'dir')
    mkdir(options.data_dir)
end

demographics_file = options.data_dir + filesep + "mics_demographics.csv";
if ~exist(demographics_file, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(demographics_file, json.mics_tutorial.demographics.url, ...
        weboptions('TimeOut', 10));
end
demographics = readtable(demographics_file, 'PreserveVariableNames', true);

thickness_file = options.data_dir + filesep + "mics_thickness.h5";
if ~exist(thickness_file, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(thickness_file, json.mics_tutorial.thickness.url);
end
thickness = h5read(thickness_file, '/thickness')';

end
