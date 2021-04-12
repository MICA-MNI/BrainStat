classdef test_utils < matlab.unittest.TestCase
    methods (Test)
        function test_apply_mask(testCase)
            % Test whether the correct data is removed by apply_mask.
            Y = rand(10, 10, 10);
            mask = randperm(10) > 5;
            
            Y_masked_brainstat = brainstat_utils.apply_mask(Y, mask, 2);
            Y_masked_expected = Y(:, mask, :);

            verifyEqual(testCase, Y_masked_brainstat, Y_masked_expected);
        end

        function test_undo_mask(testCase)
            % Test whether data is placed in the correct location by undo_mask.
            Y = rand(10, 10, 10);
            mask = randperm(20) > 10;

            Y_unmasked_brainstat = brainstat_utils.undo_mask(Y, mask, 'axis', 2);
            Y_unmasked_expected = nan(size(Y,1), numel(mask), size(Y,3));
            Y_unmasked_expected(:, mask, :) = Y;

            verifyEqual(testCase, Y_unmasked_brainstat, Y_unmasked_expected);
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