%% Remove directories
surfstat_directory = '/Users/reinder/GitHub/micaopen/surfstat_chicago';
brainstat_directory = '/Users/reinder/GitHub/BrainStat/matlab';

%% Generate data.
T = load('carsmall');
rm = structfun(@isnan,T,'Uniform',false);
f = fieldnames(T);
remove = false(100,1);
for ii = 1:numel(f)
    remove = remove | any(rm.(f{ii}),2);
end
T = structfun(@(x)x(~remove,:),T,'Uniform',false);

v1 = T.Acceleration;
v2 = T.Horsepower;
v3 = [T.Cylinders,T.Weight,T.Model_Year];
v4 = T.MPG;
c = ones(size(v1));
%% SurfStat
addpath(genpath(surfstat_directory))
which term
ss_plus = term(v3) + term(v4);
ss_minus1 = term(v3) - term(v4);
ss_minus2 = term(v3) - term([v3(:,1),v4]);

ss_model = term(c) + term(v2) + term(v3) + term(v4);
slm = SurfStatLinMod(v1,ss_model);
slm = SurfStatT(slm,c);
%% BrainStat
addpath(genpath(brainstat_directory));
which effect
bs_plus = effect(v3) + effect(v4);
bs_minus1 = effect(v3) - effect(v4);
bs_minus2 = effect(v3) - effect([v3(:,1),v4]);

bs_model = effect(c) + effect(v2) + effect(v3) + effect(v4); 
lm = BrainStatLinMod(v1,bs_model,c,slm);
%% Compare
[ ...
all(all(double(ss_plus) == bs_plus.data));
all(all(double(ss_minus1) == bs_minus1.data));
all(all(double(ss_minus2) == bs_minus2.data));
]'
