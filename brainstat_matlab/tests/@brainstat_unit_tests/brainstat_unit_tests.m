classdef brainstat_unit_tests < matlab.unittest.TestCase
    methods (Test)
        %% Test the overloads of the term/random classes.
        function term_random_overloads(testCase)
            % Initialize random data.
            v1 = rand(100,2);
            v2 = rand(100,2);
            
            % Construct fixed/random effects.
            t1 = term(v1);
            r1 = random(v1);
            
            t2 = term([v1,v2]);
            r2 = random([v1,v2]);
            
            t3 = term(v2);
            r3 = random(v2);
            
            % Test subsref.
            testCase.verifyEqual(t1.matrix, v1);
            testCase.verifyEqual(t1.names, ["v11","v12"]);
            testCase.verifyEqual(t1.v11, v1(:,1));
            testCase.verifyEqual(t1(:,1), v1(:,1));

            as_variance = @(x)reshape(x*x',[],1);
            testCase.verifyEqual(r1.variance.matrix,as_variance(v1));
            testCase.verifyEqual(r1(:,1),as_variance(v1));
            
            % Test size.
            testCase.verifyEqual(size(t1),size(v1));
            testCase.verifyEqual(size(r1 + t1), [size(v1), size(v1,1).^2 , 1]) %#ok<SZARLOG>
            
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
    end
end