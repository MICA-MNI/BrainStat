function interpolated_data = mesh_interpolate(data, source, target, options)
% MESH_INTERPOLATE    Interpolates data between two meshes.
%
%   interpolated_data = MESH_INTERPOLATE(data, source, target, options)
%   interpolates the data on the source template to the target template
%   using a nearest neighbor interpolation. Variable data must be a matrix
%   with size(data, 1) equal to the number of vertices in the source template. Source
%   and target must either be surface structures containing an n-by-3 field
%   called vertices, or a char containing the name of a valid template for
%   fetch_surface_template().
%
%   Valid Name-Value pairs:
%       data_dir:
%           Data directory for the surface templates. Used only if source
%           or target are provided as char. Defaults to
%           $HOME_DIR/brainstat_data/surface_data
%       interpolation:
%           Type of interpolation to use. Only 'nearest_neighbor' is
%           currently implemented. Defaults to 'nearest_neighbor'.

%% Handle input arguments.
arguments
    data
    source 
    target
    options.data_dir = brainstat_utils.get_brainstat_directories('surface_data_dir')
    options.interpolation (1, :) char {isValidInterpolation} = 'nearest_neighbor'
end

if isstring(source); source = char(source); end
if isstring(target); target = char(target); end

if ischar(source)
    source = fetch_template_surface(source, 'data_dir', options.data_dir, 'join', true);
end

if ischar(target)
    target = fetch_template_surface(target, 'data_dir', options.data_dir, 'join', true);
end

if size(source.vertices, 1) ~= size(data,1)
    error('BrainStat:invalidTemplate', ...
        'Number of vertices in the source template and the size of data''s first dimension are not equal.')
end

%% Interpolate.
switch lower(options.interpolation)
    case 'nearest_neighbor'
        mapping = knnsearch(source.vertices, target.vertices);
        subselect = repmat({':'}, ndims(data), 1);
        subselect{1} = mapping;
        interpolated_data = data(subselect{:});
    otherwise
        error('Invalid interpolation type. Note: this error should''ve been caught earlier, please notify the BrainStat developers.');
end
end

%% Validation function.
function isValidInterpolation(interpolation)
    valid_interpolations = {'nearest_neighbor'};
    if ~ismember(interpolation, valid_interpolations)
        add_apostrophe = cellfun(@(x) ['''' x ''''], valid_interpolations, 'Uniform', false);
        eid = 'BrainStat:InvalidInterpolation';
        msg = ['''' interpolation ''' is not a valid interpolation. Valid options are: ' ...
            strjoin(add_apostrophe, ', ') '.'];
        throwAsCaller(MException(eid, msg));
    end
end
