%% Initialize the tests
% Make sure the path is set correctly. 
git_path = '/Users/reinder/GitHub/BrainStat_MICAMNI/';

warning('off','MATLAB:rmpath:DirNotFound');
rmpath(genpath(git_path));
warning('on','MATLAB:rmpath:DirNotFound');

addpath(git_path + "matlab/surfstat_ported");
addpath(git_path + "matlab/surfstat_candidate_replacement");

%% Checks
rng(0);
v1 = rand(100,1);
r1 = random(v1);
r2 = random(rand(100,1));
t1 = term(rand(100,1));
t2 = term(rand(100,1));

plusr = r1 + r2;
plust = t1 + t1;
plusm = r1 + t1; 

minusr = r1 - r2;
minust = t1 - t1;
minusm = r1 - t1;

mpowerr = r1^5;
mpowert = t1^5;

charr = char(r1);
chart = char(t1);

[m,v] = double(r1 + t1);
m2 = double(t1);
all(m2 == m) 

empty1r = isempty(random());
empty2r = isempty(r1);
empty1t = isempty(term());
empty2t = isempty(t1);

szr = size(r1);
szt = size(t1);

r1(1)
t1(1)
