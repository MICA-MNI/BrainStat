function filename = download_OSF_files(template, options)
% Downloads files from OSF.

arguments
    template (1,:) char
    options.data_dir (1,:) char {mustBeFolder} = brainstat_utils.get_data_dir( ...
        'subdirectory', 'surfaces', 'mkdir', true);
    options.parcellation char = ''
end

[url, md5] = dataset_utils.get_OSF_url(template, options.parcellation);
filename = [options.data_dir, filesep, template, '_', options.parcellation, '.tar.gz'];
if ~exist(filename, 'file')
    websave(filename, url);
end
if ~strcmp(md5, brainstat_utils.DataHash(filename, 'file'))
    error('MD5 of file %s did not match expected MD5.', filename)
end
end
