function list = matstr2list(matstr)
% MATSTR2LIST   MATLAB string to Python list converter.
%
%   list = MATSTR2list(matstr) converts MATLAB char, cell, and string
%   arrays to Python lists. Contents of cell arrays must be readily
%   convertible to Python.

if ischar(matstr)
    list = py.list({matstr});
elseif isstring(matstr)
    list = py.list(cellstr(matstr));
elseif iscell(matstr)
    list = py.list(matstr);
else
    error('This function only accepts char, cell, and string arrays.');
end