path = fileparts(mfilename('fullpath'));
prj_file = path + string(filesep) + "BrainStat.prj";
output_file = path + string(filesep) + "BrainStat.mltbx";
matlab.addons.toolbox.packageToolbox(prj_file, output_file)