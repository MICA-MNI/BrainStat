classdef SLM < matlab.mixin.Copyable
% SLM    The core object of the BrainStat statistics module 
%   obj = SLM(model, contrast, varargin) constructs an SLM objects with the a
%   linear model specified by term/random object model, a numeric contrast, and
%   other name-value pairs (see below).
%
%   Once constructed, the model can be fitted to a dataset using obj.fit(Y), where 
%   Y is a sample-by-feature-by-variate matrix. 
%   
%   Valid name-value pairs:
%   surf: 
%       A char array containing a path to a surface, a cell/string array of the
%       aforementioned, or a loaded surface in SurfStat format. Defaults to 
%       struct(). 
%   mask:  
%       A logical vector containing true for vertices that should be kept 
%       during the analysis. Defaults to [].
%   correction:
%       A cell array containing 'rft', 'fdr', or both. If 'rft' is included, then
%       a random field theory correction will be run. If 'fdr' is included, then a 
%       false discovery rate correction will be run. Defaults to [].
%   niter:
%       Number of iterations of the fisher scoring algorithm. Defaults to 1.
%   thetalim:
%       Lower limit on variance coefficients, in sd's. Defaults 0.01
%   drlim:
%       Step of ratio of variance coefficients, in sd's. Defaults 0.1. 
%   two_tailed:
%       Whether to run one-tailed or two-tailed significance tests. Defaults to
%       true. Note that multivariate models only support two-tailed tests.
%   cluster_threshold:
%       P-value threshold or statistic threshold for defining clusters, Defaults
%       to 0.001.


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
        P
        Q
    end
    
    properties(SetAccess=protected, Hidden=true)
        V
        k
        r
        dr
        resl
        c
        ef
        sd
        dfs
        du
    end
    
    properties(SetAccess=protected, Dependent=true)
        tri
        lat 
        coord
    end

    methods
        %% Methods that are intended for user interaction. 

        function obj = SLM(model, contrast, options)
            % Constructor for the SLM class. 
            arguments
                model
                contrast {mustBeVector}
                options.surf (1,1) {brainstat_utils.validators.mustBeBrainStatSurface} = struct()
                options.mask logical {mustBeVector} = ones(size(contrast,1),1);
                options.correction string {mustBeValidCorrection} = []
                options.niter (1,1) double {mustBeInteger, mustBePositive} = 1
                options.thetalim (1,1) double {mustBePositive} = 0.01
                options.drlim (1,1) double {mustBePositive} = 0.1
                options.two_tailed (1,1) logical = true
                options.cluster_threshold (1,1) double {mustBePositive} = 0.001
            end
            
            obj.model = model;
            obj.contrast = contrast;    
            for field = fieldnames(options)'
                obj.(field{1}) = options.(field{1});
            end
           
            obj.reset_fit_parameters();
        end

        function fit(obj, Y)
            % FIT    Runs the statistics pipeline. 
            % fit(obj, Y) runs the model defined in the object. Y is a
            % (observation, region, variate) data matrix. 

            if ndims(Y) > 2 %#ok<ISMAT>
                if ~obj.two_tailed && size(Y,3) > 1
                    error('One-tailed tests are not implemented for multivariate data.');
                end
            end
            student_t_test = size(Y,3) == 1; 

            obj.reset_fit_parameters();
            if isempty(obj.mask)
                obj.mask = ~all(Y==0);
            end
            Y = brainstat_utils.apply_mask(Y, obj.mask, 2);
            
            obj.linear_model(Y);
            obj.t_test();
            obj.unmask();
            if ~isempty(obj.correction)
                obj.multiple_comparisons_corrections(student_t_test);
            end
        end

        %% Special set/get functions.
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
        
        function set.coord(obj, value)
            obj.surf.coord = value;
        end
        
        function coord = get.coord(obj)
            if contains('coord', fieldnames(obj.surf))
                coord = obj.surf.coord;
            else 
                coord = [];
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

        function multiple_comparisons_corrections(obj, student_t_test)
            % Wrapper for running and merging multiple comparisons tests.
            [P1, Q1] = obj.run_multiple_comparisons();

            if obj.two_tailed && student_t_test
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
                [P.pval, P.peak, P.clus, P.clusid] = obj.random_field_theory();
            end
            if ismember('fdr', obj.correction)
                Q = obj.fdr();
            end
        end

        function unmask(obj)
            if all(obj.mask)
                % Escape if there is no masking. 
                return
            end
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
                    P.(key1) = {P1.(key1), P2.(key1)};
                else
                    P.(key1) = struct();
                    for key2_loop = fieldnames(P1.(key1))'
                        key2 = key2_loop{1};
                        if key2 == "P" && key1 == "pval"
                            P.(key1).(key2) = brainstat_utils.one_tailed_to_two_tailed(...
                                P1.(key1).(key2), P2.(key1).(key2));
                        else
                            P.(key1).(key2) = {P1.(key1).(key2), P2.(key1).(key2)};
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
                Q = brainstat_utils.one_tailed_to_two_tailed(Q1, Q2);
            end
        end
    end
end

%% Validator functions
function mustBeValidCorrection(x)
% Validator function for multiple comparisons corrections. 
valid_corrections = {'rft', 'fdr'};
if ~all(ismember(x, valid_corrections))
    eid = 'BrainStat:notACorrection';
    msg = ['Valid corrections are: ' strjoin(valid_corrections, ', ') '.'];
    throwAsCaller(MException(eid, msg));
end
end