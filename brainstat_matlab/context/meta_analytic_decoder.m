function [pearsons_r_sort, feature_names_sort] = meta_analytic_decoder(stat_data, options)
% META_ANALYTIC_DECODER    Decodes data on the surface. 
%
%   [pearsons_r_sort, feature_names_sort] = META_ANALYTIC_DECODER(stat_data) correlates the
%   data (stat_data) on the fsaverage5 surface with the meta-analytical
%   maps from Neurosynth. Returned are the Pearson correlation values and
%   the feature names associated with those values. Note that, as this
%   function only correlates data within the cortical surface, that
%   correlation values derived from this method may differ from those
%   derived with a whole brain map. 
%
%   Valid name-value pairs are:
%       corrtype
%           'Pearson' (the default) to compute Pearson's linear
%           correlation coefficient, 'Kendall' to compute Kendall's
%           tau, or 'Spearman' to compute Spearman's rho.
%       template
%           The template surface to use, either 'fsaverage5' or
%           'fsaverage', or 'fslr32k', defaults to 'fsaverage5'.
%       interpolation
%           The type of surface to volume interpolation. Currently only
%           'nearest' (nearest neighbor interpolation) is allowed.
%       data_dir
%           The directory to store the data. Defaults to
%           $HOME_DIR/brainstat_data/neurosynth_data.
%       database
%           The database to use for decoding. Currently only 'neurosynth' is
%           allowed.
%       ascending
%           If true sort output in ascending order, otherwise in descending
%           order. Defaults to false. 
%
%   Note: if you use this function please consider citing both Neurosynth
%   and Nimare (consult their documentations for up-to-date referencing) as
%   these packages were integral to generating the data used here.

arguments
    stat_data (:, 1) {mustBeNumeric}
    options.corrtype (1,:) char = 'Pearson'
    options.template (1,:) char = 'fsaverage5'
    options.interpolation (1,:) char = 'nearest'
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('neurosynth_data_dir');
    options.database (1,:) char = 'neurosynth'
    options.ascending (1,1) logical = false
end

if options.interpolation == "nearest"
    interpolated_volume = nearest_neighbor_interpolation(options.template, stat_data);
else
    error('The only valid interpolation method is ''nearest''.');
end
if options.database ~= "neurosynth"
    error('The only valid database method is ''neurosynth''.');
end

mask = interpolated_volume ~= 0;
    
neurosynth_files = fetch_neurosynth_data(options.data_dir);
feature_names = regexp(neurosynth_files, '__[0-9a-zA-Z ]+', 'match', 'once');
feature_names = cellfun(@(x) x(3:end), feature_names, 'Uniform', false);

pearsons_r = zeros(numel(neurosynth_files), 1);
for ii = 1:numel(neurosynth_files)
    neurosynth_volume = read_volume(neurosynth_files{ii});
    mask_inf = mask & ~isinf(neurosynth_volume);
    pearsons_r(ii) = corr(interpolated_volume(mask_inf), neurosynth_volume(mask_inf), ...
        'rows', 'pairwise', 'type', options.corrtype);
end

if options.ascending
    [pearsons_r_sort, idx] = sort(pearsons_r, 'ascend');
else
    [pearsons_r_sort, idx] = sort(pearsons_r, 'descend');
end
feature_names_sort = feature_names(idx); 

end


function neurosynth_files = fetch_neurosynth_data(data_dir)
% Fetches neurosynth data files.
%
% Downloads the precomputed Neurosynth archive and extracts it into
% data_dir. Robust to interrupted downloads (which previously left a
% truncated .zip on disk and broke subsequent calls; see #341): the zip is
% verified before unzip, and the download is retried up to a small number
% of times. The zip and any partial extracted state are cleaned up if the
% download or extraction fails.

json = brainstat_utils.read_data_fetcher_json();

neurosynth_files = find_files(data_dir);

if numel(neurosynth_files) ~= json.neurosynth_precomputed.n_files
    if ~exist(data_dir, 'dir'); mkdir(data_dir); end
    disp('Downloading Neurosynth files. This may take several minutes.')

    max_attempts = 3;
    last_err = [];
    for attempt = 1:max_attempts
        zip_file = tempname(data_dir) + ".zip";
        cleanup = onCleanup(@() delete_if_exists(zip_file)); %#ok<NASGU>
        try
            websave(zip_file, json.neurosynth_precomputed.url);
            verify_zip_integrity(zip_file);
            unzip(zip_file, data_dir);
            last_err = [];
            break
        catch ME
            last_err = ME;
            warning('BrainStat:NeurosynthDownload', ...
                'Attempt %d/%d failed: %s', attempt, max_attempts, ME.message);
        end
    end

    if ~isempty(last_err)
        rethrow(last_err);
    end

    neurosynth_files = find_files(data_dir);
    if numel(neurosynth_files) ~= json.neurosynth_precomputed.n_files
        error('BrainStat:NeurosynthDownload', ...
            ['Neurosynth archive extracted but %d files were found (expected %d). ' ...
             'Try removing %s and re-running.'], ...
            numel(neurosynth_files), json.neurosynth_precomputed.n_files, data_dir);
    end
end
    function neurosynth_files = find_files(data_dir)
        if ~exist(data_dir, 'dir')
            neurosynth_files = strings(0, 1);
            return
        end
        data_dir_contents = dir(data_dir);
        data_dir_filenames = {data_dir_contents.name};
        neurosynth_files = regexp(data_dir_filenames, "Neurosynth_TFIDF.*_z_.*consistency.nii.gz", ...
            'match', 'once');
        neurosynth_files(cellfun(@isempty, neurosynth_files)) = [];
        neurosynth_files = data_dir + filesep + neurosynth_files;
    end
end

function verify_zip_integrity(zip_file)
% Sanity-checks a downloaded zip before handing it to MATLAB's unzip,
% which is unhelpfully terse when given a truncated archive. We use
% Java's ZipFile to validate the central directory without extracting
% the (multi-GB) contents.
info = dir(zip_file);
if isempty(info) || info.bytes < 1024
    error('BrainStat:NeurosynthDownload', ...
        'Downloaded zip is missing or implausibly small (%d bytes).', ...
        max(0, info.bytes));
end
try
    zf = java.util.zip.ZipFile(zip_file);
    zf.close();
catch ME
    error('BrainStat:NeurosynthDownload', ...
        'Downloaded zip failed integrity check: %s', ME.message);
end
end

function delete_if_exists(filepath)
if exist(filepath, 'file')
    delete(filepath);
end
end

function interpolated_volume = nearest_neighbor_interpolation(template, labels)
% Performs nearest neighbor interpolation using precomputed volumes. 
precomputed_data_dir = brainstat_utils.get_brainstat_directories('brainstat_precomputed_data');
volume = read_volume(precomputed_data_dir + filesep + "nn_interp_" + template + ".nii.gz");

labels_0 = [labels(:); 0];
volume(volume==0) = numel(labels_0);
interpolated_volume = labels_0(volume); 
end