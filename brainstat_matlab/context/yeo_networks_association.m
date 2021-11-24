function association = yeo_networks_association(data, options)
% YEO_NETWORKS_ASSOCIATION    Computes the association with yeo networks.
%
%   association = YEO_NETWORKS_ASSOCIATION(data, varargin) computes the 
%   mean of the datapoints within each Yeo network. Variable data must be a
%   vector with length equal to the number of vertices in the surface
%   template (default fsaverage5). 
%
%   Valid Name-Value pairs:
%       seven_networks (logical):
%           If true (default) uses the seven Yeo networks, otherwise uses
%           the seventeen network parcellation.
%       template (char):   
%           The surface template to use. Uses 'fsaverage5' by default. See
%           fetch_parcellation for a list of valid templates. 
%       data_dir (string):
%           Data directory to store the parcellation data. Defaults to
%           $HOME_DIR/brainstat_data/parcellation_data
%       reduction_operation (function_handle):
%           Function to apply to the data within each network. This
%           function should take a vector and return a scalar. Defaults to
%           @(x) mean(x, 'omitnan').

arguments
    data (:, 1) 
    options.seven_networks (1,1) logical = true
    options.template (1,:) char = 'fsaverage5'
    options.data_dir (1,1) string = brainstat_utils.get_brainstat_directories('parcellation_data_dir')
    options.reduction_operation function_handle = @(x) mean(x, 'omitnan');
end

if options.seven_networks
    n_networks = 7;
else
    n_networks = 17;
end

yeo_networks = fetch_parcellation(options.template, 'yeo', n_networks, ...
    'data_dir', options.data_dir);

association = accumarray(yeo_networks+1, data, [], options.reduction_operation);
association(1) = []; % Remove undefined (undefined = yeo_networks==0)

