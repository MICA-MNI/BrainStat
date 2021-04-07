function expression = surface_genetic_expression(pial, white, labels, volume_template, varargin)
% SURFACE_GENETIC_EXPRESSION   genetic expression of surface parcels
%
%   expression = surface_genetic_expression(pial, white, labels,
%   volume_template, varargin) computes the genetic expression for parcels
%   on the cortical surface. Pial and white are paths to pial/white surface
%   files or cell arrays containing multiple of the aforementioned. Labels
%   is the path to a parcellation file (gifti or csv) or a cell array
%   containing multipe of the aforementioned. volume_tmeplate is the path
%   to a NIFTI image to use as a template for the surface to volume
%   interpolation. Returns a region-by-gene table of expression values.
%
%   For the name-value pairs, please consult the abagen documentation
%   corresponding to your installed version of abagen. 
%
%   An equal number of pial/white surfaces and labels must be provided. If
%   parcellations overlap across surfaces (e.g. overlap at the midline),
%   then the labels are kept for the first provided surface.
%
%   This function is, essentially, a wrapper around the Python BrainStat
%   implementation. Make sure that MATLAB is using a python version (see
%   `pyenv`) that has brainstat and pyembree installed. 

%% Check Python environment
if ~context_utils.py_test_environment('brainstat')
    error('Could not find a Python environment with brainstat installed.');
end

if ~context_utils.py_test_environment('pyembree')
    error(['The Python package pyembree is required for this function. ', ...
           'You can install it with the conda package manager: ', ...
           ' `conda install -c conda-forge pyembree`.']);
end

%% Deal with input
pial = context_utils.matstr2list(pial);
white = context_utils.matstr2list(white);
if ~isa(labels, 'py.numpy.ndarray')
    labels = context_utils.matstr2list(labels);
end

p = inputParser();
p.addParameter('atlas_info', py.None, @ischar)
p.addParameter('ibf_threshold', 0.5, @isscalar)
p.addParameter('probe_selection', 'diff_stability', @ischar);
p.addParameter('donor_probes', 'aggregate', @ischar);
p.addParameter('lr_mirror', false, @islogical);
p.addParameter('exact', true, @islogical);
p.addParameter('tolerance', 2, @isscalar);
p.addParameter('sample_norm', 'srs', @ischar);
p.addParameter('gene_norm', 'srs', @ischar);
p.addParameter('norm_matched', true, @islogical);
p.addParameter('region_agg', 'donors', @ischar);
p.addParameter('agg_metric', 'mean', @ischar);
p.addParameter('corrected_mni', true, @islogical);
p.addParameter('reannotated', true, @islogical);
p.addParameter('return_counts', false, @islogical);
p.addParameter('return_donors', false, @islogical);
p.addParameter('donors', 'all', @ischar);
p.addParameter('data_dir', '', @ischar); % MATLAB based downloader - do not use py.NoneType
p.addParameter('verbose', 1, @isscalar);
p.addParameter('n_proc', 1, @isscalar);
p.parse(varargin{:});
R = p.Results;

%% Download dataset - doing this through Python causes an error.
disp('Looking for the AHBA dataset. If not found, dataset will be downloaded.');
R.data_dir = context_utils.create_genetics_dataset(R.data_dir);
R.data_dir = char(R.data_dir); % Python doesn't accept strings.

%% Run genetic decoding.
name_value_pairs = [fieldnames(R), struct2cell(R)]';
py_expression = py.brainstat.context.genetics.surface_genetic_expression( ...
    pial, white, labels, volume_template, pyargs(name_value_pairs{:})); 

% Convert Python output to MATLAB
expression = context_utils.pandas2table(py_expression);
end