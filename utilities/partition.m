% stratified n-fold cross-validation
function [cv,indices] = partition(y, nfold)

% y: vector of labels
% nfold: number of folds

cv = cvpartition(y,'kfold',nfold);
indices = zeros(size(y));
for q=1:nfold
    indices(cv.test(q)) = q;
end





end