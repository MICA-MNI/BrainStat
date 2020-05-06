function out = context_decoder(map,mask)
% CONTEXT_DECODER correlates input to template contextual maps.
%   out = CONTEXT_DECODER(map) correlates the n-by-m matrix map to several
%   neurosynth-derived terms, PET tracers, and gene expression maps. n must
%   be 64984 (conte69 surface) or 20484 (fsaverage 5 surface). 
%
%   out = CONTEXT_DECODER(map,mask) only includes data which is set to true
%   in the n-by-1 logical vector mask. 
%
%   For more information consult our <a
% href="https://brainstat.readthedocs.io/en/latest/matlab_doc/multivariate/gradientmaps.html">ReadTheDocs</a>.
%
%   See also: ADD_VISUALIZATION_FUNCTION.

%% Deal with input

% Remove masked data.
if ~exist('mask','var')
    mask = nan; 
elseif ~isvector(mask) || numel(mask) ~= size(map,1)
    error('Mask should be a vector with length equal to the number of rows in the supplied map.');
elseif ~islogical(mask)
    error('Mask must be a logical vector.');
end

%% Decode 
brainstat_path = string(fileparts(fileparts(fileparts(mfilename('fullpath')))));
data_dir = brainstat_path + filesep() + "shared" + filesep() + "contextdata" + filesep();

for type = {'pet','neurosynth'} % TODO: Add genes
    [out.(type{1}),field] = decode(map, data_dir + type + ".mat", mask);
end
out.surface = field(6:end);
end
%% Local functions. 
function [out,field] = decode(map, context_file, mask)

% Grab correct map. 
if size(map,1) == 64984
    field = 'data_conte69';
elseif size(map,1) == 20484
    field = 'data_fsa5';
else
    error(['Unknown input surface. Valid surfaces are conte69-32k '... 
        '(64984 data points) or fsaverage5 (20484 data points).']);
end
context = load(context_file, 'names', field);

% Keep only data in the mask. 
if ~isnan(mask)
    map(~mask,:) = [];
    context.(field)(~mask,:) = []; 
end

% Compute correlation.
out.r = corr(map, context.(field), 'rows', 'pairwise');
out.names = context.names; 
end