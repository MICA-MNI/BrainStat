classdef test_tutorial < matlab.unittest.TestCase
    methods (Test)
        function test_load_data(testCase)
            [image_data, demographic_data] = fetch_tutorial_data('n_subjects', 17);
            testCase.verifySize(image_data, [17, 20484]);
            testCase.verifySize(demographic_data, [17, 5]);   
            testCase.verifyEqual(fieldnames(demographic_data), {'ID2'; 'GROUP'; 'AGE'; 'HAND'; 'IQ'; 'Properties'; 'Row'; 'Variables'});
        end
    end
end