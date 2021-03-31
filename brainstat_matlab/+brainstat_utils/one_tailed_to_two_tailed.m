function p = one_tailed_to_two_tailed(p1, p2)
% Converts two one-tailed tests to a two-tailed test.
p = min(min(p1, p2) * 2, 1);
end