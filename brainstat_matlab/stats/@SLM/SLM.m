classdef SLM < matlab.mixin.Copyable

    properties
        model
        contrast
        surf
        mask
        correction
        niter
        thetalim
        drlim
        two_tailed
        cluster_threshold
    end

    properties(SetAccess=protected)
        X
        t
        df
        SSE
        coef
        V
        k
        r
        dr
        resl
        c
        ef
        sd
        dfs
        P
        Q
        du
    end
    
    properties(SetAccess=protected, Dependent=true)
        tri
        lat 
    end

    methods
        %% Methods that are intended for user interaction. 

        function obj = SLM(model, contrast, varargin)
            % Constructor for the SLM class. 

            % Deal with default inputs.
            obj.model = model;
            obj.contrast = contrast;
            
            % Parse optional arguments.
            is_correction = @(x) all(ismember(x,{'rft','fdr'}));
            p = inputParser();
            p.addParameter('surf', struct()); %TODO: Add surface validator.
            p.addParameter('mask', [], @isvector);
            p.addParameter('correction', [], is_correction);
            p.addParameter('niter', 1, @isscalar);
            p.addParameter('thetalim', 0.01, @isscalar);
            p.addParameter('drlim', 0.1, @isscalar);
            p.addParameter('two_tailed', true, @islogical);
            p.addParameter('cluster_threshold', 0.001, @isscalar)
            p.parse(varargin{:});
            for field = fieldnames(p.Results)'
                obj.(field{1}) = p.Results.(field{1});
            end

            obj.reset_fit_parameters();
        end

        function fit(obj, Y)
            % Runs the statistics pipeline using the model parameters set in the constructor. 
            %
            % Y is a (observation, region, variate) matrix. 

            if ndims(Y) > 2
                if ~obj.two_tailed && size(Y,3) > 1
                    error('One-tailed tests are not implemented for multivariate data.');
                end
            end

            obj.reset_fit_parameters();
            if ~isempty(obj.mask)
                Y = brainstat_utils.apply_mask(Y, obj.mask, 2);
            end
            obj.linear_model(Y);
            obj.t_test();
            if ~isempty(obj.mask)
                obj.unmask();
            end
            if ~isempty(obj.correction)
                obj.multiple_comparisons_corrections();
            end
        end

        %% Special set/get functions.
        function set.surf(obj, value)
            % Converts input surface to SurfStat format

            if ischar(value)
                % Assume surface is a single file.
                surf = read_surface(value);  %#ok<*PROPLC>
                obj.surf = convert_surface(surf, 'format', 'SurfStat');
            elseif isstring(value) || iscell(value)
                % Assume surface is a set of files. 
                surfaces = cellfun(@read_surface, value);
                all_surfaces = surfaces{1};
                for ii = 2:numel(surfs)
                    all_surfaces = combine_surfaces(all_surfaces, surfaces{ii}, 'SurfStat');
                end
                obj.surf = all_surfaces;
            elseif isempty(value)
                % Empty input.
                obj.surf = []; 
            elseif isstruct(value)
                % Assume input is empty or already a loaded surface.
                if isempty(fieldnames(value)) || contains('lat', fieldnames(value))
                    obj.surf = value; % Lattice format. 
                else
                    obj.surf = convert_surface(value, 'format', 'SurfStat');
                end
            else
                error('Unknown surface format.');
            end       
        end

        function set.mask(obj, value)
            % Converts input mask to logical.
            obj.mask = logical(value);
        end
        
        function set.tri(obj, value)
            obj.surf.tri = value;
        end
        
        function tri = get.tri(obj)
            if contains('tri', fieldnames(obj.surf))
                tri = obj.surf.tri;
            else 
                tri = [];
            end
        end
        
        function set.lat(obj, value)
            obj.surf.lat = value;
        end
        
        function lat = get.lat(obj)
            if contains('lat', fieldnames(obj.surf))
                lat = obj.surf.lat;
            else
                lat = [];
            end
        end
    end

    methods(Hidden = true, Access = protected)
        %% Here we put all the methods that the user should not be touching.

        function reset_fit_parameters(obj)
            % Sets all fitting parameters to empty arrays.
            obj.X = [];
            obj.t = [];
            obj.df = [];
            obj.SSE = [];
            obj.coef = [];
            obj.V = [];
            obj.k = []; 
            obj.r = []; 
            obj.dr = []; 
            obj.resl = [];
            obj.c = []; 
            obj.ef = [];
            obj.sd = []; 
            obj.dfs = [];
            obj.P = [];
            obj.Q = [];
            obj.du = [];
        end

        function multiple_comparisons_corrections(obj)
            % Wrapper for running and merging multiple comparisons tests.
            [P1, Q1] = obj.run_multiple_comparisons();

            if obj.two_tailed
                obj.t = -obj.t;
                [P2, Q2] = obj.run_multiple_comparisons();
                obj.t = -obj.t;
                obj.P = obj.merge_rft(P1, P2);
                obj.Q = obj.merge_fdr(Q1, Q2);
            else
                obj.P = P1;
                obj.Q = Q1;
            end
        end   
        
        function [P, Q] = run_multiple_comparisons(obj)
            % Runs all available multiple comparisons tests.
            P = [];
            Q = [];
            if ismember('rft', obj.correction)
                if isempty(fieldnames(obj.surf))
                    error('Random field theory requires a surface.');
                end
                P = obj.random_field_theory();
            end
            if ismember('fdr', obj.correction)
                Q = obj.fdr();
            end
        end

        function unmask(obj)
            % Changes all masked parameters to their input dimensions.
            simple_unmask_parameters = {'t', 'coef', 'SSE', 'r', 'ef', 'sd', 'dfs'};
            for key = simple_unmask_parameters
                property = obj.(key{1});
                if ~isempty(property)
                    obj.(key{1}) = brainstat_utils.undo_mask(property, obj.mask, 'axis', 2);
                end
            end

            if ~isempty(obj.resl)
                edges = mesh_edges(obj.surf);
                [~, idx] = brainstat_utils.mask_edges(edges, obj.mask);
                obj.resl = brainstat_utils.undo_mask(obj.resl, idx, 'axis', 1);
            end
        end
    end
    
    methods(Hidden = true)
        %% Debugging Tools
        function debug_set(obj, varargin)
            % Set function to circumvent protected properties. 
            warning('This function is for testing purposes only. Do not use this unless you really, really know what you''re doing.');
            for ii = 1:2:numel(varargin)
                obj.(varargin{ii}) = varargin{ii+1}; 
            end
        end
    end

    methods(Static)
        %% Static methods
        
        function P = merge_rft(P1, P2)
            % Merge two one-tailed outputs of the random_field_theory function
            % into a single structure. Both P1 and P2 are the output of the RFT function.

            P = struct();
            if isempty(P1) && isempty(P2)
                return
            end

            for key1_loop = fieldnames(P1)'
                key1 = key1_loop{1};
                if key1 == "clusid"
                    P.(key1) = [P1.(key1), P2.(key1)];
                else
                    P.(key1) = struct();
                    for key2_loop = fieldnames(P1.(key1))'
                        key2 = key2_loop{1};
                        if key2 == "P" and key1 == "pval"
                            P.(key1).(key2) = brainstat_utils.one_tailed_to_two_tailed(P1.(key1).(key2));
                        else
                            P.(key1).(key2) = [P1.(key1).(key2), P2.(key1).(key2)];
                        end
                    end
                end
            end
        end

        function Q = merge_fdr(Q1, Q2)
            % Merges output of SurfStatQ
            if isempty(Q1) && isempty(Q2)
                Q = [];
            else
                Q = brainstat_utils.one_tailed_to_two_tailed(Q1, Q2)
            end
        end
    end
end