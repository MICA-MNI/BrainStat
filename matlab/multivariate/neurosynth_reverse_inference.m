function [r,features] = neurosynth_reverse_inference(images,varargin)
% r = mica_ns_reverse_inference(images,varargin) Neurosynth conte69
% decoding
%
% Args:
%     images: A [64984,X] matrix containing the images to decode. 
%     mask: An optional mask to apply to features and input images. Trues
%     are kept. 
%     image_type: An optional string indicating the type of image to use
%         when constructing feature-based images. See
%         meta.analyze_features() for details. By default, uses reverse
%         inference z-score images.

%% Parse input
p = inputParser;
addParameter(p, 'image_type', 'association-test_z', @ischar);
addParameter(p, 'mask', nan);
parse(p, varargin{:});
R = p.Results;

%% Load images
data_path = "/host/yeatman/local_raid/reinder/neurosynth/results_on_surface/";

% Get all NeuroSynth files
all_files = dir(data_path);
filenames = {all_files.name};

% Extract those containing the requested test
filenames_test = regexp(filenames,".*" + R.image_type + "_(l|r)h.shape.gii",'match','once');
filenames_test(cellfun(@isempty,filenames_test)) = []; 

% Reshape such that left/right hemisphere are the first/second row.
filenames_test = reshape(filenames_test,2,[]);

% Load the giftis... this can take a while. 
feature_gifti = cellfun(@read_surface_data,data_path + filenames_test,'uniform',false);
%feature_gifti = cellfun(@(x)x.cdata,feature_gifti,'uniform',false);
feature_data = cell2mat(feature_gifti);

% Extract feature names
features = regexp(filenames_test(1,:),"^(.*?)_" + R.image_type,'match','once');

% Clean memory a bit
clearvars feature_gifti 
%% Run analysis

% Remove masked data.
if ~isnan(R.mask)
    feature_data(~R.mask,:) = []; 
    images(~R.mask,:) = []; 
end

% Correlate it 
r = corr(images,feature_data);     