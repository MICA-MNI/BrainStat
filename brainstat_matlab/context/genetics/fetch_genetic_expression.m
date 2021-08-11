function [expression, gene_names] = fetch_genetic_expression(atlas, n_regions, options)
% FETCH_GENETIC_EXPRESSION    fetches genetic expression data
%    [expression, gene_names] = FETCH_GENETIC_EXPRESSION(atlas, n_regions,
%    varargin) returns the genetic expression for the given atlas at the
%    resolution n_regions. gene_names contains the name of the gene
%    associated with each collumn. Genetic expression was computed using
%    the Python package abagen with default parameters on the fsaverage
%    surface.
%
%   Valid atlas / n_region pairs:
%       'schaefer': 100, 200, 300, 400, 500, 600, 800, 1000
%       'cammoun': 33, 60, 125, 250, 500
%       'glasser': 360
%   
%   Valid name-value pairs:
%       data_dir: the directory to store the expression matrices. Defaults
%           to $HOME_DIR/brainstat_data/genetics
%       seven_networks: If using the 'schaefer' atlas, uses the 7 network
%           subparcellation when true, and the 17 network parcellation when
%           false. Defaults to true.
%       overwrite: If true, overwrites existing data files. Defaults to
%           false.
%       verbose: If true, returns verbose output. Defaults to true. 
%
%   Note: this function may take a few minutes to run. Runtime gets
%   substantially longer with larger parcellations.

arguments
    atlas (1,:) char
    n_regions (1,1) double
    options.data_dir (1,1) string = brainstat_utils.get_data_dir('subdirectory', 'genetics', 'mkdir', true)
    options.seven_networks (1,1) logical = true
    options.overwrite (1,1) logical = false
    options.verbose (1,1) logical = true
end

filename_parts = {'expression', atlas, num2str(n_regions)};
if strcmp(atlas, 'schaefer')
    if options.seven_networks
        filename_parts = [filename_parts, '7Networks'];
    else
        filename_parts = [filename_parts, '17Networks'];
    end
end
filename = options.data_dir + filesep + strjoin(filename_parts, '_') + '.csv.gz';

url = get_url(atlas, n_regions, options.seven_networks);
if ~exist(filename, 'file') || options.overwrite
    if options.verbose
        disp(['Downloading atlas to ' filename{1} '.'])
    end
    websave(filename, url);
end

if options.verbose
    disp('Reading atlas from file.')
end
[expression, gene_names] = read_csv_gz(filename);
end


function url = get_url(atlas, n_regions, seven_networks)
% Grabs the URL of the requested file. 

urls = struct( ...
    'schaefer_100_7Networks', 'https://box.bic.mni.mcgill.ca/s/mOeys3OLWqFEoMV/download', ...
    'schaefer_200_7Networks', 'https://box.bic.mni.mcgill.ca/s/HJUDi4gJsDLNB1O/download', ...
    'schaefer_300_7Networks', 'https://box.bic.mni.mcgill.ca/s/D40SIfkCr9oKO0G/download', ...
    'schaefer_400_7Networks', 'https://box.bic.mni.mcgill.ca/s/cP6CPBLQWG2VlkN/download', ...
    'schaefer_500_7Networks', 'https://box.bic.mni.mcgill.ca/s/bW1WbiAeD24u5xW/download', ...
    'schaefer_600_7Networks', 'https://box.bic.mni.mcgill.ca/s/i5QmQpAWD8jH9Jo/download', ...
    'schaefer_800_7Networks', 'https://box.bic.mni.mcgill.ca/s/qHVSuhRN62tpH26/download', ...
    'schaefer_1000_7Networks', 'https://box.bic.mni.mcgill.ca/s/JJKwKk1hiGAxsML/download', ...
    'schaefer_100_17Networks', 'https://box.bic.mni.mcgill.ca/s/JtfYdNtaxJxCpwx/download', ...
    'schaefer_200_17Networks', 'https://box.bic.mni.mcgill.ca/s/JV123CNd2Qf9o6q/download', ...
    'schaefer_300_17Networks', 'https://box.bic.mni.mcgill.ca/s/JefNeipRaK6eIxj/download', ...
    'schaefer_400_17Networks', 'https://box.bic.mni.mcgill.ca/s/q0a7PT9Se6ts5DJ/download', ...
    'schaefer_500_17Networks', 'https://box.bic.mni.mcgill.ca/s/wWbZWzEdv56l2K6/download', ...
    'schaefer_600_17Networks', 'https://box.bic.mni.mcgill.ca/s/sefcvPNIcr7dMoS/download', ...
    'schaefer_800_17Networks', 'https://box.bic.mni.mcgill.ca/s/hqQzo9tb9mUGHvY/download', ...
    'schaefer_1000_17Networks', 'https://box.bic.mni.mcgill.ca/s/CG3dnJ39KrVuUAK/download', ...
    'cammoun_33', 'https://box.bic.mni.mcgill.ca/s/WUMwoGhSH5QNQ6P/download', ...
    'cammoun_60', 'https://box.bic.mni.mcgill.ca/s/DyEBiYvApUiC0LV/download', ...
    'cammoun_125', 'https://box.bic.mni.mcgill.ca/s/Q8Lu7tGzuYnm2QE/download', ...
    'cammoun_250', 'https://box.bic.mni.mcgill.ca/s/d5Iz55j46BU1hYm/download', ...
    'cammoun_500', 'https://box.bic.mni.mcgill.ca/s/hdg3cWZ3p41f6z5/download', ...
    'glasser_360', 'https://box.bic.mni.mcgill.ca/s/QCrvc44xS8dCnel/download' ...
);

if atlas == "schaefer"
    if seven_networks
        network_str = '7Networks';
    else
        network_str = '17Networks';
    end
    url = urls.(strjoin({atlas, num2str(n_regions), network_str}, '_'));
else
    url = urls.(strjoin({atlas, num2str(n_regions)}, '_'));
end
end

function [mat, gene_names] = read_csv_gz(filename)
% Read a .csv.gz file. We avoid unzipping and reading from disk as the user's
% temporary directory may be full. This does come at the cost of
% performance.

% Read file line-by-line.
file_stream = java.io.FileInputStream(filename);
inflated_stream = java.util.zip.GZIPInputStream(file_stream);
char_stream = java.io.InputStreamReader(inflated_stream);
lines = java.io.BufferedReader(char_stream);

file_contents = []; 
line = lines.readLine();
while ~isempty(line)
    file_contents = [file_contents; line]; %#ok<AGROW>
    line = lines.readLine();
end

% Get gene names.
gene_names = strsplit(char(file_contents(1,:)), ',');
gene_names(1) = []; % Remove name of row labels.

% Get data matrix.
lines_matlab = cellfun(@(x) strsplit(x, ',', 'CollapseDelimiters', false), ...
    cell(file_contents), 'Uniform', false);
file_matlab = cat(1, lines_matlab{:});
mat = cellfun(@str2double, file_matlab(2:end, 2:end));
end