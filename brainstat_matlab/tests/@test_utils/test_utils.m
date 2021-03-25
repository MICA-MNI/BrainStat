classdef test_term_random_overloads < matlab.unittest.TestCase
    methods (Test)
        function test_apply_mask(testCase)
            % Test whether the correct data is removed by apply_mask.
            Y = rand(100, 100, 100);
            mask = rand(100,1) > 0.5;
            
            Y_masked_brainstat = brainstat_utils.apply_mask(Y, mask, 2);
            Y_masked_expected = Y(:,mask,:);

            verifyEqual(testCase, Y_masked_brainstat, Y_masked_expected);
        end

        function test_undo_mask(testCase)
            % Test whether data is placed in the correct location by undo_mask.
            Y = rand(100, 100, 100);
            mask = randperm(200) > 100;

            Y_unmasked_brainstat = brainstat_utils.undo_mask(Y, mask, 'axis', 2);
            Y_unmasked_expected = zeros(100, 200, 100);
            Y_unmasked_expected(:, mask, :) = Y_unmasked_expected;

            verifyEqual(testCase, Y_masked_brainstat, Y_masked_expected);
        end

        function test_one_tailed_to_two_tailed(testCase)
            % Test whether the output value is correct.
            p1 = brainstat_utils.one_tailed_to_two_tailed(0.6, 0.2);
            verifyEqual(testCase, p1, 0.4);

            p2 = brainstat_utils.one_tailed_to_two_tailed(0.6, 0.9);
            verifyEqual(testCase, p2, 1);
        end

    end
end