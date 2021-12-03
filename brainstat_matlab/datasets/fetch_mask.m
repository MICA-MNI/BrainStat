function varargout = fetch_mask(template, options)
% FETCH_MASK    fetches masks for the midline.
%   mask = FETCH_MASK(template, varargin) fetches a cortical mask for a
%   surface template. Valid templates are civet41k, civet164k, fslr32k,
%   fsaverage5, and fsaverage. 
%
%   Valid name-value pairs:
%       'data_dir'
%           Location to store the data. Defaults to
%           $HOME_DIR/brainstat_data/surface_data.
%       'join'
%           If true, returns a single mask. If false, returns one for each
%           hemisphere. Defaults to true.
%       'overwrite'
%           If true, overwrites existing data. Defaults to false.

arguments
    template (1,1) string
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('surface_data_dir');
    options.join (1,1) logical = true
    options.overwrite (1,1) logical = false
end

template = lower(template);
mask_file = options.data_dir + filesep + template + "_mask.csv";

if ~exist(mask_file, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(mask_file, json.masks.(template).url);
end

mask = logical(readmatrix(mask_file));

if options.join
    varargout = {mask};
else
    varargout = {mask(1:end/2), mask(end/2+1:end)};
end