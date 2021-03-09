classdef test_precomputed < matlab.unittest.TestCase
    % For full details of each test, please see the tests in the 
    % Python implementation. The documentation in this version is rather
    % "lightweight". ;-)
    
    methods
        function recursive_equality(testCase, S1, S2, file)
            % Recursive tests whether the contents of all cells/struct
            % fields are equal across S1 and S2. 
            
            if ~strcmp(class(S1), class(S2))
                if isempty(S1) && isempty(S2)
                    % Getting the empties to sync across Python/MATLAB is a
                    % pain. Deal with it the easy way.
                    verifyEmpty(testCase, S1);
                    verifyEmpty(testCase, S2);
                    return;
                else
                    keyboard;
                    error('Inputs are not of the same type.');
                end
            end
            
            if isstruct(S1)
                fields = fieldnames(S1)';
                for field = fields
                    recursive_equality(testCase, S1.(field{1}), S2.(field{1}), file)
                end
            elseif iscell(S1)
                for ii = 1:numel(S1)
                    recursive_equality(testCase, S1{ii}, S2{ii}, file);
                end
            elseif isnumeric(S1)
                verifyEqual(testCase, S1, S2, 'abstol', 1e-5, ...
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
                f = fieldnames(input);
                
                if ismember('age', f)
                    input.M = 1 + term(input.age');
                elseif ismember('params', f)
                    input.M = term(input.params);
                end
                if ~ismember('surf',f)
                    input.surf = [];
                end
                if ~ismember('niter',f)
                    input.niter = 1;
                end
                if ~ismember('thetalim',f)
                    input.thetalim = 0.01;
                end
                if ~ismember('drlim', f)
                    input.drlim = 0.1;
                end
                if ismember('tri', f)
                    input.surf = struct('tri', input.tri);
                elseif ismember('lat', f)
                    input.surf = struct('lat', input.lat);
                end
                
                slm = SurfStatLinMod(input.Y, input.M, input.surf, ...
                    input.niter, input.thetalim, input.drlim);
                
                recursive_equality(testCase, slm, output, pair{1});
            end
        end

        function test_q(testCase)
            % Precomputed tests for SurfStatQ
            q_files = get_test_files('statq');
            for pair = q_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                f = fieldnames(input);
                if ismember('mask', f)
                    mask = logical(input.mask);
                else
                    mask = [];
                end
                Q = SurfStatQ(input, mask);
                Q.mask = double(Q.mask);
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
                slms = struct('slm1',struct(),'slm2',struct());
                for ii = ['1','2']
                    for field = {'X','df','SSE','coef','tri','resl','c','k','ef','sd','t'}
                        if ismember(['slm1' field{1}],f)
                            slms.(['slm', ii]).(field{1}) = input.(['slm', ii, field{1}]);
                        end
                    end
                end
                slm = SurfStatF(slms.slm1, slms.slm2);
                recursive_equality(testCase, slm, output, pair{1});
            end
        end
        
        function test_edg(testCase)
            % Test SurfStatEdg
            edg_files = get_test_files('statedg');
            for pair = edg_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                edges = SurfStatEdg(input);
                verifyEqual(testCase, ...
                    double(edges), ...
                    double(output.edg+1), ... % +1 due to the Python/MATLAB count from 0/1 difference.
                    'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
            end
        end
        
        function test_norm(testCase)
            % Test SurfStatNorm
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
                
                [Yout, Yav] = SurfStatNorm(Y, mask, subdiv);
                verifyEqual(testCase, Yout, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
                verifyEqual(testCase, Yav, output.Python_Yav, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end     
        end
        
        function test_smooth(testCase)
            % Test SurfStatSmooth
            norm_files = get_test_files('statsmo');
            for pair = norm_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                if ismember('tri', fieldnames(input))
                    surf = struct('tri', input.tri);
                else
                    surf = struct('lat', input.lat);
                end
                smoothed = SurfStatSmooth(input.Y, surf, input.FWHM);
                verifyEqual(testCase, smoothed, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end
        end
        
        function test_stand(testCase)
            % Test SurfStatStand
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
                [Y, Ym] = SurfStatStand(input.Y, mask, subdiv);

                verifyEqual(testCase, Y, output.Python_Y, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}]);
                verifyEqual(testCase, Ym, output.Python_Ym, 'abstol', 1e-5, ...
                    ['Testing failed on input file: ', pair{1}])
            end
        end
        
        function test_peakclus(testCase)
            % Test SurfStatSmooth
            peakc_files = get_test_files('statpeakc');
            for pair = peakc_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                
                clearvars slm mask thresh reselspvert edg 
                slm.t = input.t;
                slm.tri = input.tri;
                mask = logical(input.mask);
                thresh = input.thresh;
                
                f = fieldnames(input);
                if ismember('reselspvert',f)
                    reselspvert = input.reselspvert;
                else
                    reselspvert = ones(1, size(slm.t,2));
                end
                if ismember('edg', f)
                    edg = input.edg+1;
                else
                    edg = SurfStatEdg(slm);
                end
                if ismember('k', f)
                    slm.k = input.k;
                end
                if ismember('df', f)
                    slm.df = input.df;
                end
                
                [ peak, clus, clusid ] = SurfStatPeakClus(slm, mask, thresh, ...
                    reselspvert, edg);
                
                % Test equality.
                for arg = {peak, clus, clusid; output.peak, output.clus, output.clusid}
                    recursive_equality(testCase, arg{1}, arg{2}, pair{1});            
                end
            end
        end
        
        function test_p(testCase)
            statp_files = get_test_files('statp');
            for pair = statp_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                if ismember('mask', fieldnames(input))
                    mask = logical(input.mask);
                else
                    mask = [];
                end
                if ismember('clusthresh', fieldnames(input))
                    clusthresh = input.clusthresh;
                else
                    clusthresh = 0.001;
                end
                [P.pval, P.peak, P.clus, P.clusid]  = SurfStatP(input, mask, clusthresh);
                if ismember('mask', fieldnames(P.pval))
                    P.pval.mask = double(P.pval.mask);
                end
                recursive_equality(testCase, P, output, pair{1});
            end
        end
        
        function test_resels(testCase)
            statp_files = get_test_files('statresl');
            for pair = statp_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                if ismember('mask', fieldnames(input))
                    input.mask = logical(input.mask);
                end
                if ismember('resl', fieldnames(input))
                    [P.resels, P.reselspvert, P.edg] = SurfStatResels(input);
                    P.edg = double(P.edg-1);
                else
                    P.resels = SurfStatResels(input);
                end
                recursive_equality(testCase, P, output, pair{1});
            end
        end
        
        function test_statthreshold(testCase)
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
            statt_files = get_test_files('statt');
            for pair = statt_files
                input = load_pkl(pair{1});
                output = load_pkl(pair{2});
                slm = SurfStatT(input, input.contrast);
                slm = rmfield(slm,'contrast');
                recursive_equality(testCase, slm, output, pair{1});           
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
test_files = all_files(startsWith(all_files, test_name));
if isempty(test_files)
    error('Did not find any test files.')
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
    'py.NoneType', @(x) [];
    'double', @(x)x;
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