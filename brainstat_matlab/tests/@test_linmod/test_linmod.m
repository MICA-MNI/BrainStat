classdef test_linmod < matlab.unittest.TestCase
    methods (Test)
        function run_linmod_test(testCase)
            % Get all linmod test files.
            data_dir = get_test_data_dir();
            data_dir_contents = dir(data_dir);
            all_files = {data_dir_contents.name};
            linmod_files = all_files(startsWith(all_files, 'linmod'));
            
            % Run over every pair of input/output files.
            linmod_files = reshape(linmod_files,2,[]);
            for pair = linmod_files
                input = load_pkl(strjoin({data_dir, pair{1}}, filesep));
                output = load_pkl(strjoin({data_dir, pair{2}}, filesep));
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
                
                out_vars = fieldnames(output)';
                for field = out_vars
                    verifyEqual(testCase, slm.(field{1}), output.(field{1}), ...
                        'abstol', 1e-5, ...
                        ['Testing failed on input file: ', pair{1}]);
                end       
            end
        end
    end
end