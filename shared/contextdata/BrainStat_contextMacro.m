function [yeo mesul gradient f] = BrainStat_contextMacro(map,template, BPATH)
% function [r f] = BrainStat_contextMACRO(map,template, BPATH)
% spatial correlation analysis between a given brain map and Yeo-krienen
% communities, functional connectivity gradients, and mesulam laminar 
% differentiation 
% 
% inputs:   conte69 or fsa5 currently supported)
%           map - a 1 x num_vert scalar map 
%           template - a surface template in surfstat format  
%                 
%           BPATH (optional) the path for brainstat 
% outputs: 
% first version - April 22 

if vargin = 2 
    %% needs to be improved with mfilename in future :) 
    BPATH = '/Users/boris/Documents/1_github/BrainStat/'

end

% get PET data
load([BPATH '/shared/contextdata/MACRO.mat'])
map = PET.data_conte69(1,:); 
k = length(map); 
if k == 20484 
    MACRO.yeo.data = MACRO.yeo.data_fsa5; 
    MACRO.g.data = MACRO.g.data_fsa5;
    MACRO.mesul.data = MACRO.mesul.data_fsa5;
    disp('assuming data is provided on fsaverage5')
elseif k == 64984
    MACRO.yeo.data = MACRO.yeo.data_conte69; 
    MACRO.g.data = MACRO.g.data_conte69;
    MACRO.mesul.data = MACRO.mesul.data_conte69;
    disp('assuming data is provided on conte69')

else 
    disp('please specify the map either on fsa5 or conte69')
end

f=figure; 
    
