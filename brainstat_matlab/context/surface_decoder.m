function [pearsons_r_sort, feature_names_sort] = surface_decoder(stat_data, options)
% SURFACE_DECODER    Decodes data on the surface. 
%
%   [pearsons_r_sort, feature_names_sort] = SURFACE_DECODER(stat_data) correlates the
%   data (stat_data) on the fsaverage5 surface with the meta-analytical
%   maps from Neurosynth. Returned are the Pearson correlation values and
%   the feature names associated with those values. Note that, as this
%   function only correlates data within the cortical surface, that
%   correlation values derived from this method may differ from those
%   derived with a whole brain map. 
%
%   Valid name-value pairs are:
%       template
%           The template surface to use, either 'fsaverage5' or
%           'fsaverage', or 'fslr32k', defaults to 'fsaverage5'.
%       interpolation
%           The type of surface to volume interpolation. Currently only
%           'nearest' (nearest neighbor interpolation) is allowed.
%       data_dir
%           The directory to store the data. Defaults to
%           $HOME_DIR/brainstat_data/neurosynth_data.
%       database
%           The database to use for decoding. Currently only 'neurosynth' is
%           allowed.
%       ascending
%           If true sort output in ascending order, otherwise in descending
%           order. Defaults to false. 
%
%   Note: if you use this function please consider citing both Neurosynth
%   and Nimare (consult their documentations for up-to-date referencing) as
%   these packages were integral to generating the data used here.


warning('surface_decoder has been renamed to meta_analytic_decoder. Calls through surface_decoder will be removed in a future version.')

[pearsons_r_sort, feature_names_sort] = meta_analytic_decoder(stat_data, options{:});