classdef test_other < matlab.unittest.TestCase
    methods (Test)
        function test_slm_volume(testCase)
            % Test whether the correct data is removed by apply_mask.
            volume = rand(10, 10, 10) > 0.3;
            data = rand(3, sum(volume(:)));
            model = FixedEffect(rand(3,1));
            contrast = ones(3,1);

            slm = SLM(model, contrast, 'surf', volume);
            slm.fit(data)

        end


    end
end