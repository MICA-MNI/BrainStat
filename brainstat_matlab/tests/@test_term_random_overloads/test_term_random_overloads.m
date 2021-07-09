classdef test_term_random_overloads < matlab.unittest.TestCase
    methods (Test)
        function test_subsref(testCase)
            v1 = rand(100,2);
            t1 = FixedEffect(v1, "", false);
            r1 = MixedEffect(v1, [], 'add_identity', false);
            
            testCase.verifyEqual(t1.matrix, v1);
            testCase.verifyEqual(t1.names, ["v11","v12"]);
            testCase.verifyEqual(t1.v11, v1(:,1));
            testCase.verifyEqual(t1(:,1), v1(:,1));

            as_variance = @(x)reshape(x*x',[],1);
            testCase.verifyEqual(r1.variance.matrix,as_variance(v1));
            testCase.verifyEqual(r1(:,1),as_variance(v1));
        end
        
        function test_size(testCase)
            v1 = rand(100,2);
            t1 = FixedEffect(v1, "", false);
            r1 = MixedEffect(v1, [], 'add_identity', false, 'add_intercept', false);
            
            % Test size.
            testCase.verifyEqual(size(t1),size(v1));
            testCase.verifyEqual(size(r1 + t1), [size(v1), size(v1,1).^2 , 1]) %#ok<SZARLOG>
        end
        
        function test_math(testCase)
            v1 = rand(100,2);
            v2 = rand(100,2);
            t1 = FixedEffect(v1, "", false);
            t2 = FixedEffect([v1,v2], "", false);
            t3 = FixedEffect(v2, "", false);
            r1 = MixedEffect(v1, [], 'add_identity', false);
            r3 = MixedEffect(v2, [], 'add_identity', false);
            as_variance = @(x)reshape(x*x',[],1);
            
            % Test plus.
            plustt = t1 + t2;
            plusrr = r1 + r3;
            plusrt = r1 + t1;

            testCase.verifyEqual(plustt.matrix, [v1,v2]);
            testCase.verifyEqual(plusrr.variance.matrix, [as_variance(v1),as_variance(v2)]);
            testCase.verifyEqual(plusrt.mean.matrix, v1);
            testCase.verifyEqual(plusrt.variance.matrix,as_variance(v1));

            % Test minus.
            minustt = t2 - t1;
            minusrr = r3 - r1;
            minusrt = r1 - t1;

            testCase.verifyEqual(minustt.matrix, v2);
            testCase.verifyEqual(minusrr.variance.matrix, as_variance(v2));
            testCase.verifyEqual(minusrt.variance.matrix, as_variance(v1));
            testCase.verifyEmpty(minusrt.mean.matrix);

            % Test mtimes.
            timestt = t1 * t3;
            timesrr = r1 * r3;
            %timesrt = r1 * t1;

            act_times = reshape(v1 .* permute(v2,[1,3,2]),100,4);
            testCase.verifyEqual(timestt.matrix, act_times);
            testCase.verifyEqual(timesrr.variance.matrix, as_variance(v2) .* as_variance(v1));
            %testCase.verifyEqual(timesrt.variance.matrix, ???);

            % Test mpower.
            powert = t1 ^ 2;
            powerr = r1 ^ 2;

            act_power = reshape(v1 .* permute(v1,[1,3,2]),100,4);
            act_power(:,2) = [];
            testCase.verifyEqual(act_power, powert.matrix);
            
            testCase.verifyEqual(as_variance(v1).^2, powerr.variance.matrix);
        end
        
        function test_identity_detection(testCase)
            r1 = MixedEffect(rand(3,1), [], 'add_identity', false);
            r2 = MixedEffect(1, [], 'name_ran', 'test_identity');
            I = eye(3);
            
            r12 = r1 + r2;
            r21 = r2 + r1;
            testCase.verifyEqual(r12.variance.matrix(:,2), I(:));
            testCase.verifyEqual(r21.variance.matrix(:,2), I(:));
        end
    end
end
