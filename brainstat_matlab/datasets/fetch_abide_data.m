function [thickness, demographics] = fetch_abide_data(options)
% FETCH_ABIDE_DATA    fetches thickness and demographic ABIDE data
%   [thickness, demographics] = FETCH_ABIDE_DATA(varargin) fetches ABIDE
%   cortical thickness and demographics data. The following name-value
%   pairs are allowed:
%
%   'data_dir'
%       The directory to save the data. Defaults to
%       $HOME/brainstat_data/abide_data
%   'sites'
%       The sites to keep subjects from. Defaults to all sites.
%   'keep_control'
%       If true, keeps control subjects, defaults to true.
%   'keep_patient'
%       If true, keeps patients, defaults to false.
%   'overwrite'
%       If true, overwrites older files. Defaults to false.

arguments
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('abide_data_dir')
    options.sites string = ""
    options.keep_control (1,1) logical = true
    options.keep_patient (1,1) logical = true
    options.overwrite (1,1) logical = false
end

if ~exist(options.data_dir, 'dir')
    mkdir(options.data_dir)
end

summary_spreadsheet = options.data_dir + filesep + "summary_spreadsheet.csv";
if ~exist(summary_spreadsheet, 'file') || options.overwrite
    json = brainstat_utils.read_data_fetcher_json();
    websave(summary_spreadsheet, json.abide_tutorial.summary_spreadsheet.url);
end

demographics = readtable(summary_spreadsheet, 'VariableNamingRule', 'preserve');
demographics = select_subjects(demographics, options.sites, options.keep_patient, options.keep_control);

thickness = zeros(size(demographics,1), 81924);
keep_rows = ones(size(demographics,1), 1, 'logical'); 
hemi = ["left", "right"];

for ii = 1:size(demographics,1)
    for jj = 1:numel(hemi)
        filename = options.data_dir + filesep + ...
            sprintf("%s_%s_thickness.txt", num2str(demographics.SUB_ID(ii)), hemi(jj));
        if ~exist(filename, 'file') || options.overwrite
            url = thickness_url("native_rms_rsl_tlink_30mm_" + hemi{jj}, demographics.FILE_ID{ii});
            try
                websave(filename, url)
            catch err
                delete(filename); % An empty file is created on failure.
                if ismember(err.identifier, ["MATLAB:webserivces:UnknownHost", "MATLAB:webservices:HTTP404StatusCodeError"])
                    keep_rows(ii) = false;
                    continue
                else
                    rethrow(err)
                end
            end
        end
        thickness(ii, 1 + (jj-1) * 40962 : jj * 40962) = dlmread(filename);   
    end
end

thickness = thickness(keep_rows, :);
demographics = demographics(keep_rows, :);

end

function demographics = select_subjects(demographics, sites, keep_patient, keep_control)
% Get a subselection of subjects.
if ~keep_patient
    demographics = demographics(demographics.DX_GROUP ~= 1, :);
end

if ~keep_control
    demographics = demographics(demographics.DX_GROUP ~= 2, :);
end

if ~isempty(sites)
    demographics = demographics(ismember(demographics.SITE_ID, sites), :);
end
end

function url = thickness_url(derivative, identifier)
    url = sprintf('https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/civet/thickness_%s/%s_%s.txt', derivative, identifier, derivative);
end