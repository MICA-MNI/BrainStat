function tbl = pandas2table(pandas)
% PANDAS2TABLE    Converts a pandas dataframe to a MATLAB table
%
%   tbl = pandas2table(pandas) converts the pandas dataframe pandas to a
%   MATLAB table. Note that MATLAB's `pyenv` most use a Python version with
%   pandas installed.

vals = double(pandas.to_numpy);
names_pystr = cell(py.list(pandas.columns));
names = cellfun(@char, names_pystr, 'Uniform', false);
tbl = array2table(vals, 'VariableNames', names);
end