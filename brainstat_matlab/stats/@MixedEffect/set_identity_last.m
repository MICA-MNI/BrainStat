function obj = set_identity_last(obj)

if isempty(obj.variance)
    return
end

n = sqrt(size(obj.variance.matrix,1));
I = eye(n);
idx = find(all(obj.variance.matrix == I(:)), 1);
if isempty(idx)
    return
end

if idx ~= numel(obj.variance.names)
    obj.variance.names([idx, end]) = obj.variance.names([end, idx]);
    obj.variance.matrix(:, [idx, end]) = obj.variance.matrix(:, [end, idx]);
end