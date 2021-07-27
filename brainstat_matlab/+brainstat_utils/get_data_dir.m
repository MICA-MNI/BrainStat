function data_dir = get_data_dir(options)
% Fetches the default BrainStat data directory. 

arguments
    options.subdirectory char = ''
    options.mkdir (1,1) logical = false
end

data_dir = [brainstat_utils.get_home_dir, filesep, 'brainstat_data', filesep, options.subdirectory];
if options.mkdir && ~exist(data_dir, 'dir')
    mkdir(data_dir)
end
end