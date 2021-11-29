function profiles = read_histology_profile(options)
% READ_HISTOLOGY_PROFILE    Reads histology profiles into memory
%
%   profiles = READ_HISTOLOGY_PROFILE(varargin) downloads and reads the
%   histology profiles. Valid name value pairs are 'data_dir' with a
%   directory to store the data, 'template' with 'fsaverage, 'fsaverage5',
%   or 'fslr32k', and 'overwrite' with true of false to overwrite old
%   files. Default values are: data_dir = HOME_DIRECTORY/histology_data,
%   template = 'fsaverage', and overwrite = false.
%
%   See also COMPUTE_MPC.

arguments
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('bigbrain_data_dir');
    options.template (1,1) string = "fsaverage"
    options.overwrite (1,1) logical = false
end

if contains(options.template, 'civet')
    civet_template = options.template;
    options.template = "fsaverage";
else
    civet_template = '';
end

if ~isfolder(options.data_dir)
    mkdir(options.data_dir)
end
histology_file = options.data_dir + filesep + "histology_" + options.template + ".h5";

if ~isfile(histology_file) || options.overwrite
    download_histology_profiles(histology_file, options.template)
end

if isempty(civet_template)
    profiles = double(h5read(histology_file, "/" + options.template));
else
    warning('CIVET histology profiles were not included with BrainStat. Interpolating from fsaverage.')
    profiles_fsaverage = double(h5read(histology_file, "/fsaverage"));
    profiles = mesh_interpolate(profiles_fsaverage, 'fsaverage', civet_template);
end
end

function download_histology_profiles(histology_file, template)
urls = struct('fsaverage', 'https://box.bic.mni.mcgill.ca/s/znBp7Emls0mMW1a/download', ...
        'fsaverage5', 'https://box.bic.mni.mcgill.ca/s/N8zstvuRb4sNcSe/download', ...
        'fslr32k', 'https://box.bic.mni.mcgill.ca/s/6zKHcg9xXu5inPR/download');
    
websave(histology_file, urls.(template));
end

