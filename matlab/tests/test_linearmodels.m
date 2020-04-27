%% Initialize the tests
% Make sure the path is set correctly. 
git_path = '/Users/reinder/GitHub/BrainStat_MICAMNI/';

warning('off','MATLAB:rmpath:DirNotFound');
rmpath(genpath(git_path));
warning('on','MATLAB:rmpath:DirNotFound');

addpath(git_path + "matlab/surfstat_ported");
cd(git_path)

% Get some random variables. 
rng(0);
v1 = rand(100,1);
v2 = rand(100,1);
v3 = rand(100,1);
y = rand(100,1);

%% Show the displays
cd(git_path + "matlab/temporary_directories/surfstat_candidate_replacement");
fixed_object = term(v1);
random_object = random(v2);

disp('Fixed object display:');
fixed_object
fprintf('\n\n\n')
disp('Random object display:');
random_object

%% First lets show that the new random/term classes are equivalent.
% Get a p-value for a pseudo-random BrainStat model.
cd(git_path + "matlab/temporary_directories/surfstat_candidate_replacement");
M = 1 + term(v1) + random(v2) + term(v3) + I; 
slm = SurfStatLinMod(y,M);
slm = SurfStatT(slm,v1);
p = SurfStatP(slm);

% Get the same p-value with a pseudo-random SurfStat model
cd(git_path + "matlab/temporary_directories/surfstat");
clear M random_object fixed_object
M2 = 1 + term(v1) + random(v2) + term(v3) + I; 
slm2 = SurfStatLinMod(y,M2);
slm2 = SurfStatT(slm2,v1);
p2 = SurfStatP(slm2);

p2.P == p.P