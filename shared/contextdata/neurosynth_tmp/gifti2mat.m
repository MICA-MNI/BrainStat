D = dir('/Users/reinder/GitHub/BrainStat_MICAMNI/shared/contextdata/neurosynth_tmp');
files = {D.name};
files(~endsWith(files,'.shape.gii')) = [];
files = reshape(files,2,[])';


for ii = 1:size(files,1)
    gii_l = gifti(files{ii,1});
    gii_r = gifti(files{ii,2});
    
    data(:,ii) = [gii_l.cdata;gii_r.cdata];
end

names = regexprep(files(:,1),'_association.*','');
save('/Users/reinder/GitHub/BrainStat_MICAMNI/shared/contextdata/neurosynth.mat','data','names','-v7.3')