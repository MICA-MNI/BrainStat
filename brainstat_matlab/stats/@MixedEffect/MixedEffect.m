classdef (InferiorClasses = {?FixedEffect}) MixedEffect
%Makes a fixed and random effects term into a mixed effects model.
%
% Usage: model = MixedEffect( ran [,fix [,strran [,strfix [,ranisvar]]] );
%
% Internally a random consists of two terms that describe linear models for
% the mean and the variance. The variables for the variance term are
% symmetric matrices stored as column vectors. If n is the number of rows
% in the mean term, then n^2 is the number of rows in the variance term.
%
% ran    = term, or anything that can be converted into a term, for the
%          variance. The single variable for the variance term is
%          vec(double(ran)*double(ran)'). If ran is a scalar s, then it is
%          vec(identity matrix)*s^2 whose size matches that of the other
%          term in a binary operator, and its name is 'Is^2' or 'I' if s=1.
% fix    = term, or anything that can be converted into a term, for the
%          mean, empty if empty or absent. ran and fix must have the same
%          number of rows, except when one is a scalar.
% strran = string for name of ran. If empty or absent, 'ran' is used. If
%          ran is a scalar, then 'I' is used.
% strfix = cell array of strings for names for the variables in fix. If
%          empty or absent, chosen by FixedEffect(fix).
% ranisvar = 1 if ran is already a term for the variance.
% If ran is a random, model=ran. With no arguments, model is empty.
%
% Let model.mean be the mean term, and model.variance be the variance term.
% The following operators are overloaded for random models m1 and m2:
%
% m = m1 + m2: m.mean = m1.mean + m2.mean,  and
%              m.variance = m1.variance + m2.variance.
%
% m = m1 - m2: m.mean = m1.mean - m2.mean,  and
%              m.variance = m1.variance - m2.variance.
%
% m = m1 * m2: m.mean = m1.mean * m2.mean,  and
%              m.variance = m1.variance * m2.variance +
%              m1.variance * sum_{i<=j} MixedEffect( (m2.mean_i+m2.mean_j)/2 ) +
%              m1.variance * sum_{i<j}  MixedEffect( (m2.mean_i-m2.mean_j)/2 ) +
%              sum_{i<=j} MixedEffect( (m1.mean_i+m1.mean_j)/2 ) * m2.variance +
%              sum_{i<j}  MixedEffect( (m1.mean_i-m1.mean_j)/2 ) * m2.variance,
%
%              where _i indicates variable i. m_.mean_i is divided by its
%              maximum absolute value so the variables have comparable
%              size, and the factor of 1/2 produces nicer printout.
%              These last four extra sums are equivalent to
%
%              sum_{i<=j} vec( m1.mean_i * m1.mean_j' ) * m2.variance +
%              sum_{i<=j} vec( m2.mean_i * m2.mean_j' ) * m1.variance,
%
%              which allow for an interaction between fixed effects and
%              random effects with arbitrary variance matrices that is
%              invariant to any linear transformation of the fixed effects.
%              The advantage of the former parameterisation is that if its
%              coefficients are positive then the variance matrix of the
%              observations is positive definite. The reason is that
%              element-wise products of positive definite matrices are
%              positive definite. However enforcing positivity has two bad
%              side effects: we lose invariance to any linear
%              transformation of the fixed effects, and it restricts the
%              range of positive definite matrices in the model.
%              Specifically, if r is the correlation between two fixed
%              effects (crossed with a random effect) with standard
%              deviations sd1<=sd2, then |r|<=sd1/sd2. In other words, if
%              the sd's are very different, then the correlaton can't be
%              too big.
%
% m = m1 ^ k:  product of m1 with itself, k times.
%
% Algebra: commutative, associative and distributive rules for random a,b,c
%    a + b = b + a
%    a * b = b * a
%    (a + b) + c = a + (b + c)
%    (a * b) * c = a * (b * c)
%    (a + b) * c = a * c + b * c   only if a,b,c are purely fixed or random
%    a + 0 = a
%    a * 1 = a
%    a * 0 = 0
% However note that
%    a + b - c =/= a + (b - c) =/= a - c + b
% If t1,t2 are terms:
%    MixedEffect(t1 * t2)  =  MixedEffect(t1) * MixedEffect(t2), but
%    MixedEffect(t1 + t2) =/= MixedEffect(t1) + MixedEffect(t2), since the LHS is one term
%
% The following functions are overloaded for random model:
% char(model)    = [char(model.mean), char(model.variance)].
% double(model)  = [double(model.mean), double(model.variance)].
% size(model)    = [size(model.mean), size(model.variance)].
% isempty(model) = 1 if model is empty and 0 if not.
    
    
    properties
        variance
        mean
    end
    
    methods
        % Constructor. 
        function obj = MixedEffect(ran, fix, varargin)
            
            % Parse input
            p = inputParser;
            addOptional(p,'add_identity', true, @islogical)
            addOptional(p,'add_intercept', true, @islogical);
            addOptional(p,'name_ran', 'ran', @(x) ischar(x) || isempty(x));
            addOptional(p,'name_fix', [], @(x) ischar(x) || iscell(x) || isstring(x) || isempty(x))
            addOptional(p,'ranisvar', [], @(x)(islogical(x) || x == 0 || x == 1) && numel(x) == 1);
            parse(p, varargin{:});
            R = p.Results;
            if nargin < 2 
                fix = [];
            end
            
            % Deal with odd inputs.
            if nargin == 0
                return
            elseif isa(ran,'MixedEffect')
                obj = ran;
                warning('First input argument is already a random term; returning first input argument.');
                return
            end
            
            % If no name provided, extract from input. 
            if isempty(R.name_ran)
                R.name_ran = inputname(1);
            end
            if isempty(R.name_fix) && nargin > 1
                R.name_fix = inputname(2);
            end
            
            % Set the mean and variance.
            obj.mean = FixedEffect(); % Initialize in case it isn't set. 
            obj.variance = FixedEffect(); % Initialize in case it isn't set.
            if ~isempty(ran)
                if R.ranisvar
                    % If the random term is already a variance term, simply set
                    % the variance. 
                    obj.variance = FixedEffect(ran, R.name_ran, false);
                else
                    if numel(ran) == 1
                        % If the random term is a scalar, set the name to
                        % I.
                        if ran == 1
                            R.name_ran = 'I';
                        else
                            R.name_ran = "I" + ran + "^2";
                        end
                    end
                    
                    % Compute the variance and set it. 
                    v = double(ran)*double(ran)';
                    obj.variance = FixedEffect(v(:), R.name_ran, false);
                end
            end
                
            if ~isempty(fix)
                % Set the fixed effect. 
                obj.mean = FixedEffect(fix, R.name_fix, R.add_intercept);
            end
            
            if R.add_identity 
                obj = obj + MixedEffect(1, [], 'name_ran', 'I', 'add_identity', false);
            end
            obj = obj.set_identity_last();
        end
    end
end
