function check_categorical_variables(x, names)
% Checks whether categorical variables were provided as numerics.
%
%   CHECK_CATEGORICAL_VARIABLES(x, names) tests whether input array x is numeric
%   but may be intended to be categorical. Throws a warning if the number of
%   unique values in a column of x is less than the minimum of 5 or the size of
%   the column minus 1.
%
%   Note: this function uses a very simple heuristic and is not guaranteed to
%   catch all cases.

arguments
    x (:, :) 
    names (1, :) string = ""
end

if isscalar(x) || ~isnumeric(x)
    return
end

if all(names == "") || all(names == "?")
    names = "Column " + size(x, 2);
end

categorical_warning_threshold = min(5, size(x, 1) - 1);

for ii = 1:size(x, 2)
    unique_numbers = unique(x(:, ii));
    if length(unique_numbers) <= categorical_warning_threshold
        warning([names{ii} ' has ' num2str(length(unique_numbers)) ' unique values ' ...
        'but was supplied as a numeric (i.e. continuous) variable. Should it be a ' ...
        'categorical variable? If yes, the easiest way to provide categorical ' ...
        'variables is to convert numerics to a string array.'])
    end
end
end

