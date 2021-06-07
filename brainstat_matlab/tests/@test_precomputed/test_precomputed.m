classdef test_precomputed < matlab.unittest.TestCase
    % For full details of each test, please see the tests in the 
    % Python implementation. The documentation in this version is rather
    % "lightweight". ;-)
    
    methods
        function recursive_equality(testCase, S1, S2, file, abstol, reltol)
            % Recursive tests whether the contents of all cells/struct
            % fields are equal across S1 and S2. 
            if ~exist('abstol','var')
                abstol= 1e-5;
            end
            if ~exist('reltol','var')
                reltol = 1e-10;
            end   
            
            if ~strcmp(class(S1), class(S2))
                if isempty(S1) && isempty(S2)
                    % Getting the empties to sync across Python/MATLAB is a
                    % pain. Deal with it the easy way.
                    verifyEmpty(testCase, S1);
                    verifyEmpty(testCase, S2);
                    return;
                else
                    error('Inputs are not of the same type.');
                end
            end
            
            if isstruct(S1)
                fields = fieldnames(S1)';
                for field = fields
                    recursive_equality(testCase, S1.(field{1}), S2.(field{1}), file, abstol, reltol)
                end
            elseif iscell(S1)
                for ii = 1:numel(S1)
                    recursive_equality(testCase, S1{ii}, S2{ii}, file, abstol, reltol);
                end
            elseif isnumeric(S1)
                verifyEqual(testCase, S1, S2, 'abstol', abstol, 'reltol', reltol, ...
                    ['Testing failed on input file: ', file]);
            end
        end
    end
    
    methods (Test)
        function test_linmod(testCase)
            % Precomputed tests for SurfStatLinMod
            linmod_files = get_test_files('linmod');
            for pair = linmod_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                if isvector(input.M)
                    input.M = input.M(:);
                end
                slm = input2slm(input);
                
                slm.linear_model(input.Y);
                slm_output = slm2struct(slm, fieldnames(output)); 

                recursive_equality(testCase, slm_output, output, pair{1});
            end
        end

        function test_q(testCase)
            % Precomputed tests for SurfStatQ
            q_files = get_test_files('statq');
            for pair = q_files(:,9)
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                output = output.Q;
                
                slm = input2slm(input);
                Q = slm.fdr();
                recursive_equality(testCase, Q, output, pair{1});
            end
        end

        function test_f(testCase)
            % Precomputed tests for SurfStatF
            f_files = get_test_files('statf');
            for pair = f_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                f = fieldnames(input);
                slms = struct();
                for ii = 1:2
                    ii_reverse = ii*-1+3; % ii=1; ii_reverse=2 and vice versa.
                    f_idx = f(contains(f, ['slm', num2str(ii_reverse)]));
                    input_idx = rmfield(input, f_idx);
                    field_idx = fieldnames(input_idx);
                    for jj = 1:numel(field_idx)
                        input_idx = renameStructField(input_idx, field_idx{jj}, ...
                            replace(field_idx{jj}, ['slm', num2str(ii)], '')); 
                    end
                    slms.(['slm', num2str(ii)]) = input2slm(input_idx); 
                end

                slms.slm_f = slms.slm1.f_test(slms.slm2);
                slm_output = slm2struct(slms.slm_f, fieldnames(output));
                recursive_equality(testCase, slm_output, output, pair{1});
            end
        end
        
        function test_edg(testCase)
            % Test mesh_edges
            edg_files = get_test_files('statedg');
            for pair = edg_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                edges = mesh_edges(input);
                verifyEqual(testCase, ...
                    double(edges), ...
                    double(output.edg+1), ... % +1 due to the Python/MATLAB count from 0/1 difference.
                    'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
            end
        end
        
        function test_norm(testCase)
            % Test mesh_normalize
            norm_files = get_test_files('statnor');
            for pair = norm_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                Y = input.Y;
                subdiv = 's';
                if ismember('mask', fieldnames(input))
                    mask = logical(input.mask);
                else
                    mask = [];
                end
                if isvector(output.Python_Yav)
                    % Dimension of 1D vector needs to be transposed.
                    output.Python_Yav = output.Python_Yav';
                end
                
                [Yout, Yav] = mesh_normalize(Y, mask, subdiv);
                verifyEqual(testCase, Yout, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
                verifyEqual(testCase, Yav, output.Python_Yav, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end     
        end
        
        function test_smooth(testCase)
            % Test mesh_smooth
            smo_files = get_test_files('statsmo');
            for pair = smo_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                if ismember('tri', fieldnames(input))
                    surf = struct('tri', input.tri); % Python starts at 0.
                else
                    surf = struct('lat', input.lat);
                end
                smoothed = mesh_smooth(input.Y, surf, input.FWHM);
                verifyEqual(testCase, smoothed, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end
        end
        
        function test_stand(testCase)
            % Test mesh_standardize
            stand_files = get_test_files('statsta');
            for pair = stand_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                subdiv = 's';
                if ismember('mask', fieldnames(input))
                    mask = logical(input.mask);
                else
                    mask = [];
                end
                [Y, Ym] = mesh_standardize(input.Y, mask, subdiv);

                verifyEqual(testCase, Y, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
                verifyEqual(testCase, Ym, output.Python_Ym, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end
        end
        
        function test_peakclus(testCase)
            % Test SurfStatPeakClus
            peakc_files = get_test_files('statpeakc');
            for pair = peakc_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                clearvars slm mask thresh reselspvert edg 
                slm = input2slm(input); 
                thresh = input.thresh;
                
                if iscell(input.df)
                    input.df = cell2mat(input.df);
                end
                if ~isempty(input.reselspvert)
                    reselspvert = input.reselspvert;
                else
                    reselspvert = ones(1, size(slm.t,2));
                end
                f = fieldnames(input);
                if ismember('edg', f)
                    edg = input.edg+1;
                else
                    edg = mesh_edges(slm.surf);
                end
                
                [ peak, clus, clusid ] = slm.peak_clus(thresh, ...
                    reselspvert, edg);
                
                % Test equality.
                for arg = {peak, clus, clusid; output.peak, output.clus, output.clusid}
                    recursive_equality(testCase, arg{1}, arg{2}, pair{1});            
                end
            end
        end
        
        function test_p(testCase)
            % Test SurfStatP.
            statp_files = get_test_files('statp_');
            for pair = statp_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                slm = input2slm(input);
                
                P = struct();
                [P.pval, P.peak, P.clus, P.clusid]  = slm.random_field_theory();
                if ismember('mask', fieldnames(P.pval))
                    P.pval.mask = double(P.pval.mask);
                end
                
                % Deal with difference of empty struct in P / empty cell in
                % output. 
                if isempty(P.pval.C)
                    P.pval = rmfield(P.pval, 'C');
                end
                P.pval = rmfield(P.pval, 'mask');

                recursive_equality(testCase, P, output, pair{1});
            end
        end
        
        function test_resels(testCase)
            % Test SurfStatResels.
            statp_files = get_test_files('statresl');
            for pair = statp_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                slm = input2slm(input);
                P = struct();
                
                if ~isempty(slm.tri)
                    slm.debug_set('tri', slm.tri + 1); % Python starts at 0.
                end
                
                if ismember('resl', fieldnames(input))
                    [P.resels, P.reselspvert, P.edg] = slm.compute_resels();
                    P.edg = double(P.edg-1);
                else
                    P.resels = slm.compute_resels();
                end
                recursive_equality(testCase, P, output, pair{1});
            end
        end
        
        function test_statthreshold(testCase)
            % Test stat_threshold.
            thresh_files = get_test_files('thresh');
            for pair = thresh_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                
                for field = fieldnames(input)'
                    if iscell(input.(field{1}))
                        input.(field{1}) = cell2mat(input.(field{1}));
                    end
                end
    
                [P.peak_threshold, P.extent_threshold, P.peak_threshold_1, ...
                        P.extent_threshold_1, P.t, P.rho ] = stat_threshold(...
                       input.search_volume, ...
                       input.num_voxels, ...
                       input.fwhm, ...
                       input.df, ...
                       input.p_val_peak, ...
                       input.cluster_threshold, ...
                       input.p_val_extent, ...
                       input.nconj, ...
                       input.nvar, ...
                       [], ...
                       [], ...
                       input.nprint);
                 P.t = P.t'; 
                 recursive_equality(testCase, P, output, pair{1});
            end
        end 
        
        function test_ttest(testCase)
            % Test SurfStatT
            statt_files = get_test_files('statt');
            for pair = statt_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                slm = input2slm(input);
                slm.t_test();
                slm_output = slm2struct(slm, fieldnames(output));
                recursive_equality(testCase, slm_output, output, pair{1});           
            end
        end
    end
end


function test_files = get_test_files(test_name)
% Gets the files for a particular test. Returns them with the input/output
% of the same test along the first dimension, different tests along the
% second.

data_dir = get_test_data_dir();
data_dir_contents = dir(data_dir);

all_files = {data_dir_contents.name};

% See if the updated tests exist.
test_files = all_files(startsWith(all_files, ['x', test_name]));
if isempty(test_files)
    test_files = all_files(startsWith(all_files, test_name));
    if isempty(test_files)
        error('Could not find test files.')
    end
end
test_files = reshape(test_files, 2, []);
test_files = data_dir + string(filesep) + test_files;
end

function test_data_dir = get_test_data_dir()
% Returns the path to the test data directory.
filepath = fileparts(mfilename('fullpath'));
brainstat_dir = fileparts(fileparts(fileparts(filepath)));

test_data_dir = strjoin({brainstat_dir, 'extern', 'test-data'}, filesep);
end    

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
    'py.int', @(x) double(x);
    'py.numpy.uint8', @(x) double(x);
    'py.numpy.uint16', @(x) double(x);
    'py.NoneType', @(x) [];
    'double', @(x)x;
    'logical', @(x)x;
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

function s = slm2struct(slm, names)
% Converts the requested SLM fields to a struct.
s = struct();
for ii = 1:numel(names)
    s.(names{ii}) = slm.(names{ii});
end
end

function slm = input2slm(input)
% Converts the .pkl input to an SLM object. 
f = fieldnames(input);
if any(f == "mask")
    input.mask = logical(input.mask);
else
    input.mask = [];
end

if any(any(f == ["tri", "lat"]))
    if any(f == "coord")
        coord = input.coord;
        input = rmfield(input, "coord");
    else
        coord = [];
    end
    if any(f == "tri")
        name = 'tri';
    else
        name = 'lat';
    end
    input.surf = struct(name, input.(name), 'coord', coord);
    input = rmfield(input, name);
end

if any(f == "M")
    model = input.M;
    input = rmfield(input, "M");
else
    model = 1;
end

if any(f == "contrast")
    contrast = input.contrast;
    input = rmfield(input, "contrast");
else
    contrast = 1;
end

if any(f == "age")
    model= 1 + FixedEffect(input.age');
    input = rmfield(input, 'age');
end

if any(f == "params")
    model = FixedEffect(input.params);
    input = rmfield(input, 'params');
end

if any(f == "clusthresh")
    input.cluster_threshold = input.clusthresh;
    input = rmfield(input, 'clusthresh');
end

for field = ["Y", "colnames", "reselspvert", "edg", "thresh"]
    try 
        input = rmfield(input, field{1});
    catch err
        if err.identifier == "MATLAB:rmfield:InvalidFieldname"
            continue
        else
            rethrow(err)
        end
    end       
end
    
slm = SLM(model, contrast);
parameters = [fieldnames(input), struct2cell(input)]';
slm.debug_set(parameters{:});
end
