classdef FixedEffect
%Makes a vector, matrix or structure into a term in a linear model.
%
% Usage: t = FixedEffect( x [, str] );
%
% Internally a term consists of a cell array of strings for the names of
% the variables in the term, and a matrix with one column for each name,
% which can be accessed by char and double (see below). 
% 
% If x is a matrix of numbers, t has one variable for each column of x. 
% If x is a cell array of strings or a matrix whose rows are strings, t  
%      is a set of indicator variables for the unique strings in x. 
% If x is a structure, t has one variable for eaxh field of x. 
% If x is a scalar, t is the constant term. It is expanded to the length
%       of the other term in a binary operator.
% If x is a term, t=x. With no arguments, t is the empty term. 
% 
% str is a cell array of strings for the names of the variables. If absent,
% the names for the four cases above are either 'x' (followed by 1,2,... if
% x is a matrix), the unique strings in x, the fields of x, num2str(x) if x
% is a scalar, or '?' if x is an algebraic expression.
%
% Term t can be subscripted and returns a numeric vector or matrix, e.g.
%    obj.f      = variable with name 'f'.
%    t(1,3:5) = matrix whose columns are variables 1,3,4,5.
%
% The following operators are overloaded for terms t1 and t2:
%    t1 + t2 = {variables in t1} union {variables in t2}.
%    t1 - t2 = {variables in t1} intersection complement {variables in t2}.
%    t1 * t2 = sum of the element-wise product of each variable in t1 with
%              each variable in t2, and corresponds to the interaction 
%              between t1 and t2.
%    t ^ k   = product of t with itself, k times. 
%
% Algebra: commutative, associative and distributive rules apply to terms:
%    a + b = b + a
%    a * b = b * a
%    (a + b) + c = a + (b + c)
%    (a * b) * c = a * (b * c)
%    (a + b) * c = a * c + b * c
%    a + 0 = a
%    a * 1 = a
%    a * 0 = 0
% However note that 
%    a + b - c =/= a + (b - c) =/= a - c + b
% 
% The following functions are overloaded for term t:
%    char(t)         = cell array of the names of the variables.
%    double(t)       = matrix whose columns are the variables in t, i.e.
%                      the design matrix of the linear model.
%    size(t [, dim]) = size(double(t) [, dim]).
%    isempty(t)      = 1 if t is empty and 0 if noobj.

    properties
        names
        matrix
    end

    methods
        function obj = FixedEffect(x, names, add_intercept)
            arguments
                x = []
                names string = ""
                add_intercept logical = true
            end
            
            % If no input.
            if nargin == 0
                obj.names = [];
                obj.matrix = []; 
            
            % If input is already a term.
            elseif isa(x,'FixedEffect')
                obj = x;
            
            % If input is a cell array of char arrays, or a string array. 
            elseif iscellstr(x) || isstring(x)
                obj.names = string(unique(x))';
                obj.matrix = obj.names' == x;
            
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
                obj = FixedEffect(1, 'intercept', false) + obj;
            end
        end
    end
end
