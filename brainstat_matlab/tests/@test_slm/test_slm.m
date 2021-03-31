classdef test_slm < matlab.unittest.TestCase
    % This file is intended as a temporary test until MATLAB/Python SLM
    % tests are set up. 
    
    methods (Test)
        function test_slm_basic(testCase)
            samples = 10;
            predictors = 1; 
            
            model = term(1) + term(rand(samples, predictors));
            contrast = rand(samples,predictors);
            Y = rand(samples, 10, predictors);
            
            slm = SLM(model, contrast, 'correction', {'fdr'});
            slm.fit(Y);
            verifyNotEmpty(testCase, slm.Q);
        end   
    end
end