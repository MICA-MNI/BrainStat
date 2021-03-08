function contents = load_pkl(pkl_file)
% Loads Python .pkl files into MATLAB.
fid = py.open(pkl_file, 'rb');
data = py.pickle.load(fid);
contents = recursive_pkl_conversion(data);
end

function mat_data = recursive_pkl_conversion(pkl_data)
% Recursively converts the contents of a .pkl file to MATLAB. 
conversions = {
    'py.dict', @(x) struct(x);
    'py.list', @(x) cell(x);
    'py.numpy.ndarray', @(x) double(x);
    'py.numpy.int64', @(x) double(x)
    'py.str', @(x) char(x);
};

selection = ismember(conversions(:,1), class(pkl_data));
fun = conversions{selection,2};
try
    mat_data = fun(pkl_data);
catch err
    if class(pkl_data) == "py.numpy.ndarray"
        % Assume its a numpy nd-array containing strings
        mat_data = cell(pkl_data.tolist);
    else
        rethrow(err);
    end
end

% Recurse through structure/cell arrays.
if isstruct(mat_data)
    f = fieldnames(mat_data);
    for ii = 1:numel(f)
        mat_data.(f{ii}) = recursive_pkl_conversion(mat_data.(f{ii}));
    end
elseif iscell(mat_data)
    for ii = 1:numel(mat_data)
        mat_data{ii} = recursive_pkl_conversion(mat_data{ii});
    end
end
end