function packages_exist = py_test_environment(varargin)
% TEST_ENVIRONMENT    Tests for a correct Python environment.
%
%   packages_exist = TEST_ENVIRONMENT() checks whether a Python
%   environment with the requested packages exists. Packages must be
%   provided as char arrays in separate arguments.

packages_exist = true;
for ii = 1:numel(varargin)
    package_info = py.importlib.util.find_spec(varargin{ii});
    packages_exist = packages_exist && ~isa(package_info, 'py.NoneType');
    if ~packages_exist
        break
    end
end
