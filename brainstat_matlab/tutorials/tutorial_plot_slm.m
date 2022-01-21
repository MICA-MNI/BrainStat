function obj = tutorial_plot_slm(slm, surfaces, options)
% Tutorial function for plotting SLM data.
%
%   obj = TUTORIAL_PLOT_SLM(slm, surfaces, options) plots the data in an slm
%   to the cortical surfaces included in cell array surfaces. The following
%   name-value pairs are allowed:
%
%   mask
%       A logical vector with `false` for all vertices to exclude.
%   plot_t 
%       Plot the t-values, defaults to true.
%   plot_clus
%       Plot the clusterwise p-values, defaults to false. 
%   plot_peak
%       Plot the peak p-values, defaults to false.
%   plot_fdr
%       Plot the vertexwise p-values, defaults to false.
%   alpha
%       The upper limit of p-values to plot, defaults to 0.05.
%   t_colorlimits
%       The lower/upper limit of t-values to plot, defaults to [-6, 4].

arguments
    slm
    surfaces
    options.mask (:,1) = nan
    options.plot_t (1,1) logical = true
    options.plot_clus (1,1) logical = false
    options.plot_peak (1,1) logical = false
    options.plot_fdr (1,1) logical = false
    options.alpha (1,1) double = 0.05
    options.t_colorlimits (1,2) double = [-6, 4]
end

to_plot = [];
labels = {};
colormaps = {};
colorlimits = [];

if options.plot_t
    to_plot = [to_plot, slm.t(:)];
    labels = [labels, {'t-values'}];
    colormaps = [colormaps; {[parula; .7 .7 .7]}];
    colorlimits = [colorlimits; options.t_colorlimits];
end

if options.plot_clus
    to_plot = [to_plot, slm.P.pval.C(:)];
    labels = [labels, {{'Cluster', 'p-values (RFT)'}}];
    colormaps = [colormaps; {[flipud(autumn); .7 .7 .7]}];
    colorlimits = [colorlimits; 0, options.alpha];
end
    
if options.plot_peak
    to_plot = [to_plot, slm.P.pval.P(:)];
    labels = [labels, {{'Peak', 'p-values (RFT)'}}];
    colormaps = [colormaps; {[flipud(autumn); .7 .7 .7]}];
    colorlimits = [colorlimits; 0, 0.05];
end

if options.plot_fdr
    to_plot = [to_plot, slm.Q(:)];
    labels = [labels, {{'Vertexwise', 'p-values (FDR)'}}];
    colormaps = [colormaps; {[flipud(autumn); .7 .7 .7]}];
    colorlimits = [colorlimits; 0, 0.05];
end

if ~isnan(options.mask)
    to_plot(~options.mask, :) = inf;
end

obj = plot_hemispheres(...
    to_plot,  ...
    {surfaces{1}, surfaces{2}}, ...
    'labeltext', labels ...
    );

obj.colormaps(colormaps);
obj.colorlimits(colorlimits);
end