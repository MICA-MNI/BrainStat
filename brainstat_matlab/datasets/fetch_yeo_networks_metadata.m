function [network_names, colormap] = fetch_yeo_networks_metadata(n_regions)
% FETCH_YEO_NETWORKS_METADATA    fetches names and colormaps of the yeo networks
%
% [network_names, colormap] = FETCH_YEO_NETWORKS_METADATA(n_regions) returns the
% names and colormaps of the yeo networks. The number of regions must be either
% 7 or 17.

switch n_regions
    case 7
        network_names = [
            "Visual"
            "Somatomotor"
            "Dorsal Attention"
            "Ventral Attention"
            "Limbic"
            "Frontoparietal"
            "Default"];
        colormap = [
            120    18   134
            70   130   180
            0   118    14
            196    58   250
            220   248   164
            230   148    34
            205    62    78] / 255;
    case 17
        network_names = [
            "Visual A"
            "Visual B"
            "Somatomotor A"
            "Somatomotor B"
            "Dorsal Attention A"
            "Dorsal Attention B"
            "Salience / Ventral Attention A"
            "Salience / Ventral Attention B"
            "Limbic A"
            "Limbic B"
            "Frontoparietal C"
            "Frontoparietal A"
            "Frontoparietal B"
            "Temporal Parietal"
            "Default C"
            "Default A"
            "Default B"];
        colormap = [
            120    18   134
            255     0     0
            70   130   180
            42   204   164
            74   155    60
            0   118    14
            196    58   250
            255   152   213
            220   248   164
            122   135    50
            119   140   176
            230   148    34
            135    50    74
            12    48   255
            0     0   130
            255   255     0
            205    62    78] / 255;
    otherwise
        error('Valid values for n_regions are 7 and 17.')
end