function directory = get_brainstat_directories(name, options)
arguments
    name (1,:) char
    options.mkdir (1,1) logical = true
end

switch lower(name)
    case 'brainstat_dir' 
        directory = string(fileparts(fileparts(mfilename('fullpath'))));
    case 'brainstat_precomputed_data'
        directory = brainstat_utils.get_brainstat_directories('brainstat_dir') ...
            + filesep + "data";
    case 'brainstat_data_dir' 
        directory = brainstat_utils.get_home_dir('s') + filesep + "brainstat_data";
    case 'abide_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "abide_data";
    case 'bigbrain_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "bigbrain_data";
    case 'gradient_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "gradient_data";
    case 'genetic_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "genetic_data";
    case 'neurosynth_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "neurosynth_data";
    case 'parcellation_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir')...
            + filesep + "parcellation_data";
    case 'surface_data_dir'
        directory = brainstat_utils.get_brainstat_directories('brainstat_data_dir') ...
            + filesep + "surface_data";
    otherwise
        error('Unknown directory name.');
end

if options.mkdir && ~exist(directory, 'dir')
    mkdir(directory)
end 
end