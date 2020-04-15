classdef effect
    
    properties(SetAccess = private)
        data
        type
        names
    end
    
    methods
        function obj = effect(values,varargin)            
            % Parse input
            p = inputParser;
            addParameter(p,'names',[])
            addParameter(p,'type','fixed');
            parse(p,varargin{:})
            R = p.Results;
            
            % Check for correct names input.
            if isempty(R.names) && size(values,2) == 1
                % If no names provided and single vector.
                R.names = string(inputname(1));
                if isempty(R.names)
                    R.names = "v1"; 
                end
            elseif isempty(R.names) && size(values,2) > 1
               % If no names provided and a matrix. 
                R.names = "v" + (1:size(values,2));
            elseif ischar(R.names)
               % If names is a character array
                R.names = string(R.names);
            elseif ~iscell(R.names) && ~isstring(R.names)
                % If names is a table. 
                error('Names parameter must be a cell array, string array, or character array.');
            end
            
            % Check for duplicate names
            if numel(unique(R.names)) ~= numel(R.names)
                warning('Found duplicate variable names, appending _varX to all variable names.');
                R.names = R.names(:) + "_var" + (1:numel(R.names))';
            end
            
            % Check the number of variable names. 
            if numel(R.names) ~= size(values,2)
                error('The number of variable names must equal the number of variables.');
            end
            
            % Set properties. 
            obj.data = values;
            obj.names = R.names(:); 
            obj.type = R.type; 
        end
    end
end
