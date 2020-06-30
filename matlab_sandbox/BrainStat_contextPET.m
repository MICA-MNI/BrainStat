function [r f] = BrainStat_contextPET(map,template, BPATH)
% function [r f] = BrainStat_contextPET(map,template)
% spatial correlation analysis between a given brain map and multiple
% neurotransmitter PET maps, based on JuSpace data that were 
% mapped to cortical surfaces. 
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
load([BPATH '/shared/contextdata/PET.mat'])
map = PET.data_conte69(1,:); 
k = length(map); 
if k == 20484 
    PET.data = PET.data_fsa5; 
    disp('assuming data is provided on fsaverage5')
elseif k == 64984
    PET.data = PET.data_conte69; 
    disp('assuming data is provided on conte69')

else 
    disp('please specify the map either on fsa5 or conte69')
end

num_feat = size(PET.data,1); 

r = corr(map', PET.data');
f = figure, 
    set(gcf,'Color','white','InvertHardcopy','off');
    a(1) = axes('position',[0.2 0.15 0.4 0.8])
    barh(r)
    axis([-1.1 1.1 0 num_feat+1])
    box off
    yticks(1:length(PET.name))
    yticklabels(PET.name)
    xlabel('Spatial Correlation')
    v=length(map);
    vl=1:(v/2);
    vr=vl+v/2;
    t=size(template.tri,1);
    tl=1:(t/2);
    tr=tl+t/2;
    for i = 1:length(PET.name)
        m = 0.075
        a(1+i) = axes('position',[0.65 0.1+(i*m) m m])
        trisurf(template.tri(tl,:),...
                template.coord(1,vl),template.coord(2,vl),template.coord(3,vl),...
                double(PET.data(i,vl)),'EdgeColor','none');
        view(-90,0); 
                daspect([1 1 1]); axis tight; camlight; axis vis3d off;
                lighting phong; material dull; shading interp;
    end
      l = 1+i;     
    for i = 1:length(PET.name)
        m = 0.075
        a(l+i) = axes('position',[0.75 0.1+(i*m) m m])
        trisurf(template.tri(tl,:),...
                template.coord(1,vl),template.coord(2,vl),template.coord(3,vl),...
                double(PET.data(i,vl)),'EdgeColor','none');
        view(90,0); 
                daspect([1 1 1]); axis tight; camlight; axis vis3d off;
                lighting phong; material dull; shading interp;
    end
        
    
        






