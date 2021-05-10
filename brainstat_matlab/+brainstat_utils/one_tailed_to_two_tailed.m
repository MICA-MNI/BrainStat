function p = one_tailed_to_two_tailed(p1, p2)
% ONE_TAILED_TO_TWO_TAILED    converts one-tailed tests to two-tailed.
%   p = ONE_TAILED_TO_TWO_TAILED(p1, p2) converts two one-tailed tests with
%   p-values p1 and p2 to a single two-tailed p-value. 

p = min(min(p1, p2) * 2, 1);
end