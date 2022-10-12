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
%       A char array containing a path to a surface, a cell/string array of
%       the aforementioned, a loaded surface in SurfStat format, a template
%       name, or a 3D volume array containing 0s for excluded voxels and
%       non-zero numerics otherwise. . Valid template names are:
%       fsaverage3', 'fsaverage4', 'fsaverage5' (Y), 'fsaverage6',
%       'fsaverage' (Y), 'fslr32k' (Y), 'civet41k' (Y), 'civet164k' (Y). If
%       the template name is marked with (Y), then a list of Yeo-7 network
%       labels is returned with the table in slm.P.peak. Defaults to
%       struct().
%   mask:  
%       A logical vector containing true for vertices that should be kept 
%       during the analysis. Defaults to [].
%   correction:
%       A cell array containing 'rft', 'fdr', or both. If 'rft' is included, then
%       a random field theory correction will be run. If 'fdr' is included, then a 
%       false discovery rate correction will be run. Defaults to [].
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
%   data_dir:
%       Location to store parcellation files. Used only if surf is a
%       template name
%       Defaults to $HOME_DIR/brainstat_data


    properties
        model
        contrast
        surf
        surf_name
        mask
        correction
        niter
        thetalim
        drlim
        two_tailed
        cluster_threshold
        data_dir
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
                contrast
                options.surf = struct()
                options.mask logical = 1;
                options.correction string {mustBeValidCorrection} = []
                options.thetalim (1,1) double {mustBePositive} = 0.01
                options.drlim (1,1) double {mustBePositive} = 0.1
                options.two_tailed (1,1) logical = true
                options.cluster_threshold (1,1) double {mustBePositive} = 0.001
                options.data_dir (1,:) string = ""
            end
            
            obj.model = model;
            obj.contrast = contrast;
            [options.surf, obj.surf_name] = obj.parse_surface(options.surf);
            for field = fieldnames(options)'
                obj.(field{1}) = options.(field{1});
            end
            obj.niter = 1;
           
            obj.reset_fit_parameters();
        end

        function fit(obj, Y)
            % FIT    Runs the statistics pipeline. 
            % fit(obj, Y) runs the model defined in the object. Y is a
            % (observation, region, variate) data matrix. 

            if numel(obj.mask) == 1
                obj.mask = ones(size(Y,2), 1, 'logical');
            end
            
            if ndims(Y) > 2 %#ok<ISMAT>
                if ~obj.two_tailed && size(Y,3) > 1
                    error('One-tailed tests are not implemented for multivariate data.');
                end
            end

            if size(Y, 3) > 3 && contains(obj.correction, 'rft')
                error('RFT corrections have not been implemented for more than 3 variates.')
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

        function [sk, ku] = qc(obj, Y, options)
            % QC    Quality check. 
            % qc(obj, Y) runs quality check of the data. Y is a
            % (observation, region, variate) data matrix. feat is dimension 
            % of variate to qc - default 1 (univariate), if multivariate 
            % must specify. v is to specify vertex or parcel number - 
            % default to all. If true (default), histo will output a 
            % histogram of the residuals for vertices v. 
            %
            % Outputs vertexwise kurtosis and skewness 

            % Options
            arguments
                obj
                Y
                options.feat (1,1) double {mustBePositive} = 1
                options.v (1,:) double {mustBeNumeric} = [1:size(Y, 2)]
                options.histo (1,1) logical = true
                options.qq (1,1) logical = true
            end

            if numel(obj.mask) == 1
                obj.mask = ones(size(Y,2), 1, 'logical');
            end
            
            if ndims(Y) > 2
                Y = squeeze(Y(:, :, feat));
            end
            
            % Histogram of the residuals
            if options.histo 
                f = figure;
                    histogram(Y(:, options.v) - ...
                        (obj.X*obj.coef(:, options.v)));
                    set(gca,'box','off');
                    title('Histogram of the residuals');
            end

            % qqplot of the residuals
            if options.qq 
                f = figure;
                    qq = qqplot(Y(:, options.v) - ...
                        (obj.X*obj.coef(:, options.v)));
                    qq(1).Marker = 'o';
                    qq(1).MarkerFaceColor = 'black';
                    qq(1).MarkerEdgeColor = 'white';
                    qq(1).MarkerSize = 8.88;
                    qq(3).LineStyle = '-';
                    qq(3).LineWidth = 2.8;
                    qq(3).Color = 'black';
            end

            % Characterize distribution based on two statistical 
            % moments at each vertex
            sk = zeros(size(Y, 2), 1);
            ku = zeros(size(Y, 2), 1);
            
            for ii = 1:size(Y, 2)
                sk(ii) = skewness(Y(:, ii) - (obj.X*obj.coef(:, ii)));
                ku(ii) = kurtosis(Y(:, ii) - (obj.X*obj.coef(:, ii)));
            end
            sk(isnan(sk)) = -inf; ku(isnan(ku)) = -inf; 
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
                % Make sure output format is the same as two-tailed.
                if ~isempty(P1)
                    for field = {'peak', 'clus', 'clusid'}
                        if field == "clusid"
                            obj.P.(field{1}) = {obj.P.(field{1})};
                        else
                            for field2 = fieldnames(obj.P.(field{1}))'
                                obj.P.(field{1}).(field2{1}) = {obj.P.(field{1}).(field2{1})};
                            end
                        end
                    end
                end
                obj.Q = Q1;
            end

            obj.surfstat_to_brainstat_rft();
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

        function surfstat_to_brainstat_rft(obj)
            % Converts SurfStat structure arrays of peak and clus to tables
            % with the Yeo networks included. 
            %
            % Yeo networks are included only if the surface was defined by
            % a char array.
            
            if ismember('rft', obj.correction)
                f = fieldnames(obj.P);
                if ismember('peak', f)
                    if ismember(obj.surf_name, {'fsaverage', 'fsaverage5', 'fslr32k', 'civet41k', 'civet164k'})
                        if obj.data_dir ~= ""
                            yeo7 = fetch_parcellation(obj.surf_name, 'yeo', 7, 'data_dir', obj.data_dir);
                        else
                            yeo7 = fetch_parcellation(obj.surf_name, 'yeo', 7);
                        end
                        yeo_names = ["Undefined"; fetch_yeo_networks_metadata(7)];
                        for ii = 1:numel(obj.P.peak.t)
                            yeo7_index = yeo7(obj.P.peak.vertid{ii});
                            obj.P.peak.yeo7{ii} = yeo_names(yeo7_index + 1);
                        end
                    end
                end
                for field = ["peak", "clus"]
                    if ismember(field{1}, f)
                        for ii = 1:numel(obj.P.(field{1}).P)
                            one_tail_array = structfun(@(x) x{ii}, obj.P.(field{1}), 'Uniform', false);
                            P_field_tmp.(field{1}){ii} = struct2table(one_tail_array);
                            P_field_tmp.(field{1}){ii} = sortrows(P_field_tmp.(field{1}){ii}, 'P', 'ascend');
                        end
                    end
                end
                obj.P.peak = P_field_tmp.peak;
                obj.P.clus = P_field_tmp.clus;
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
        
        function [surf_out, surf_name] = parse_surface(surf)
            % Parses input surfaces. 
            surf_name = '';
            if ischar(surf)
                if ismember(surf, {'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6', ...
                    'fsaverage', 'fslr32k', 'civet41k', 'civet164k'})
                    surf_name = surf;
                    surf_out = fetch_template_surface(surf_name, 'join', true);
                    surf_out = io_utils.convert_surface(surf_out, 'format', 'SurfStat');
                end
            elseif isnumeric(surf) || islogical(surf)
                surf_out = struct('lat', surf ~= 0);
            elseif numel(surf) > 1
                surf_out = io_utils.combine_surfaces(surf{1}, surf{2}, 'SurfStat');
            elseif ~isempty(fieldnames(surf))
                surf_out = io_utils.convert_surface(surf, 'format', 'SurfStat');
            else
                surf_out = struct();
            end
        end

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
                    P.clusid = {P1.(key1); P2.(key1)};
                    continue
                end
                P.(key1) = struct();
                for key2_loop = fieldnames(P1.(key1))'
                    key2 = key2_loop{1};
                    if key1 == "pval"
                        P.(key1).(key2) = brainstat_utils.one_tailed_to_two_tailed(...
                            P1.(key1).(key2), P2.(key1).(key2));
                    else
                        P.(key1).(key2) = {P1.(key1).(key2); P2.(key1).(key2)};
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