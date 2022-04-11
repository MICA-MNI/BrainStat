classdef FixedEffect
% FIXED_EFFECT    Class for fixed effects.
%
%   obj = FixedEffect(x) creates a class for fixed effects. The data,
%   x, may be represented in several formats:
%
%   - If x is a matrix of numbers, obj has one variable for each column of x. 
%   - If x is a cell array of chars or a string array, obj is a set of 
%       categorical variables for the unique strings in x. 
%   - If x is a structure, obj has one variable for eaxh field of x. 
%   - If x is a scalar, obj is the constant term. It is expanded to the length
%       of the other term in a binary operator.
%   - If x is a FixedEffect, t=x. 
%   - With no arguments, obj is the empty term. 
% 
%   obj = FixedEffect(x, names) creates a class for fixed effects and adds the
%   names of the variables to the class. names is a cell array of chars or a
%   string array containing the names of the variables. If absent, the names for
%   the four cases above are either 'x' (followed by 1,2,... if x is a matrix),
%   the unique strings in x, the fields of x, num2str(x) if x is a scalar, or '?'
%   if x is an algebraic expression.
%
%   The following operators are overloaded for terms obj1  and obj2:
%       obj1  + obj2 = {variables in obj1} union {variables in obj2}.
%       obj1  - obj2 = {variables in obj1} intersection complement {variables in obj2}.
%       obj1  * obj2 = sum of the element-wise product of each variable in obj1  with
%              each variable in obj2, and corresponds to the interaction 
%              between obj1  and obj2.
%       obj ^ k   = product of obj with itself, k times. 
%
%   Algebra: commutative, associative and distributive rules apply to terms:
%       a + b = b + a
%       a * b = b * a
%       (a + b) + c = a + (b + c)
%       (a * b) * c = a * (b * c)
%       (a + b) * c = a * c + b * c
%       a + 0 = a
%       a * 1 = a
%       a * 0 = 0
%   However note that 
%       a + b - c =/= a + (b - c) =/= a - c + b
% 
%   The following functions are overloaded for term t:
%       char(t)         = cell array of the names of the variables.
%       double(t)       = matrix whose columns are the variables in t, i.e.
%                      the design matrix of the linear model.
%       size(t [, dim]) = size(double(t) [, dim]).
%       isempty(t)      = 1 if obj is empty and 0 if not.

    properties
        names % Names of the variables.
        matrix % Sample-by-feature data matrix. 
    end

    methods
        function obj = FixedEffect(x, names, add_intercept, run_categorical_check)
            arguments
                x = [] % Input data.
                names string = "" % Names of input variables.
                add_intercept logical = true % If true, include an intercept term.
                run_categorical_check logical = true % If true, check whether categorical terms are added correctly.
            end
            
            % If no input.
            if nargin == 0
                obj.names = [];
                obj.matrix = []; 
                return
            end
            
            if islogical(x)
                x = double(x);
            end
            
            % If input is already a term.
            if isa(x,'FixedEffect')
                obj = x;
            
            % If input is a cell array of char arrays, or a string array. 
            elseif iscellstr(x) || isstring(x)
                obj.names = string(unique(x))';
                obj.matrix = obj.names == x(:);
            
            % If input is a character array of two dimensions. Could
            % perhaps be merged with the previous if-statement if the
            % definition of obj.names can be homogenized.
            elseif ischar(x) && ismatrix(x)
                obj.names = string(unique(x,'rows'));
                obj.matrix = obj.names' == string(x); 
            
            % If x is numeric and not a scalar (probably the most common
            % usage)
            elseif isnumeric(x) && ~isscalar(x) 
                % Grab the term name. 
                if names == ""
                    names = inputname(1);
                    if isempty(names)
                        names = '?';
                    end
                end
                names = string(names); 
                
                % Set the term name
                if numel(names) == size(x,2)
                    obj.names = names(:)';
                elseif numel(names) == 1
                    obj.names = names + (1:size(x,2));
                end
                
                % Set the matrix
                obj.matrix = double(x); 
                if run_categorical_check
                    brainstat_utils.check_categorical_variables(obj.matrix, obj.names);
                end
            
            % If x is an numeric scalar. 
            elseif isnumeric(x) && isscalar(x)
                if exist('names', 'var')
                    obj.names = string(names);
                else
                    obj.names = string(x);
                end
                obj.matrix = double(x);
            
            % If x is a structure.
            elseif isstruct(x)
                obj.names = fieldnames(x)';
                obj.matrix = [];
                for ii = 1:length(obj.names)
                    obj.matrix = [obj.matrix double(x.(obj.names{ii}))];
                end                   
            end
            
            % Add an intercept if none exists.
            if add_intercept && ~any(all(obj.matrix == 1))
                obj = FixedEffect(1, 'intercept', false, false) + obj;
            end
        end
    end
end

