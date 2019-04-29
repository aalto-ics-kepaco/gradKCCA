% demonstration of the preimage 
clear

rng('shuffle')
% generate data
n = 300;
p = 5;
q = p;

[X,Y] = generate_algebraic_relations(n,p,q,2,2);
%[X,Y] = generate_trigonometric_relations(n,p,q,3);

% a = -1; b = 1;
% X = a + (b-a).*rand(n,p);
% Y = a + (b-a).*rand(n,q);

X = zscore(X); Y = zscore(Y);
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1; test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

% linear kernel
% Kxtrain = Xtrain * Xtrain';
% Kytrain = Ytrain * Ytrain';
% Kxtest = Xtest * Xtrain';
% Kytest = Ytest * Ytrain';

% % polynomial kernel of degree 2 (homogeneous)
% Kxtrain = centre_kernel((Xtrain * Xtrain').^2);
% Kytrain = centre_kernel((Ytrain * Ytrain').^2);
% Kxtest = center_test_kernel((Xtest * Xtrain').^2,(Xtrain * Xtrain').^2);
% Kytest = center_test_kernel((Ytest * Ytrain').^2,(Ytrain * Ytrain').^2);

% gaussian kernel
Kxtrain = centre_kernel(rbf_kernel2(Xtrain)); 
Kytrain = centre_kernel(rbf_kernel2(Ytrain)); 
Kxtest = center_test_kernel(rbf_kernel2(Xtest,Xtrain),rbf_kernel2(Xtrain));
Kytest = center_test_kernel(rbf_kernel2(Ytest,Ytrain),rbf_kernel2(Ytrain));

[alpha,beta,rho_kcc_train] = kcca_gep(Kxtrain,Kytrain,0.02,0.02,1);

% test kernel canonical correlation
rho_kcc_test = correlation2(Kxtest * alpha, Kytest * beta);

% compute approximate u and v (only for linear kernel)
% u2 = Xtrain' * alpha;
% v2 = Ytrain' * beta;

% compute u and v
%[u] = preimage_kcca(Xtrain,alpha,4);
%[v] = preimage_kcca(Ytrain,beta,4);

%[u, v]








