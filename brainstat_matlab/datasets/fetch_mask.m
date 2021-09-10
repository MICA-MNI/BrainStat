function varargout = fetch_mask(template, options)
% FETCH_CIVET_MASK    fetches masks for the midline.
%   mask = FETCH_CIVET_MASK(template, varargin) fetches a mask for the CIVET41k or
%   CIVET164k templates. Other templates are yet to be added. 

arguments
    template (1,1) string
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('surface_data_dir');
    options.join (1,1) logical = true
    options.overwrite (1,1) logical = false
end

mask_file = options.data_dir + filesep + template + "_mask.csv";

if ~exist(mask_file, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(mask_file, json.masks.(template).url);
end

mask = logical(dlmread(mask_file));

if options.join
    varargout = {mask};
else
    varargout = {mask(1:end/2), mask(end/2+1:end)};
end