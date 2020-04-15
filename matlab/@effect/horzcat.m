function out = horzcat(obj,varargin)
% Horizontal concatenation is identical to repeated plus
out = obj;
for ii = 1:numel(varargin)
    out = out + varargin{ii};
end
