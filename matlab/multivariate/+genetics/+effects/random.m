function [t,p,mean_slope] = random(X, Y, group)

names = string(unique(group));
for ii = 1:numel(names)
    X_group = X(group == names(ii));
    X_group = [ones(size(X_group)),X_group];
    betas(ii,:) = regress(Y(group == names(ii)), X_group);
end

mean_slope = mean(betas(:,2));
[~,p,~,stats] = ttest(betas(:,2),0);
t = stats.tstat; 
end