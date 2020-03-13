%% Demonstration of gradKCCA

% This is a demonstration script to introduce "Large-Scale Sparse Kernel
% Canonical Correlation Analysis", gradKCCA, published in the 
% Proceedings of the 36 th International Conference on Machine
% Learning, Long Beach, CA, USA, PMLR 81, 2019. 

%% Generate Simulated Data
clear

% generate data
n = 800;
p = 10;
q = p;

%[X,Y] = generate_algebraic_relations(n,p,q,2,4);
[X,Y] = generate_trigonometric_relations(n,p,q,3);

X = zscore(X); Y = zscore(Y);
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1; test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

correct_u = zeros(p,1); correct_u(1:2) = 1;
correct_v = zeros(q,1); correct_v(1:2) = 1;

%% the relations can be visualised by, for example:
figure
plot(Xtrain(:,1)+Xtrain(:,2),Ytrain(:,1)+Ytrain(:,2),'*')
axis off

%% gradKCCA
degree = 2; % 1 - linear kernel % 2 - quadratic kernel
hyperparams.normtypeX = 1; % norm constraint on u
hyperparams.normtypeY = 1; % norm constraint on v
hyperparams.Cx = 1; % value of the norm constraint
hyperparams.Cy = 1; % value of the norm constraint
hyperparams.Rep = 15; % number of repetitions
hyperparams.eps = 1e-10; % stoppin criterion
hyperparams.degree1 = degree; % degree of the polynomial kernel
hyperparams.degree2 = degree; % degree of the polynomial kernel

[u1,v1] = gradKCCA(Xtrain,Ytrain,1,hyperparams);
rho_train1 = corre((Xtrain * u1).^degree, (Ytrain * v1).^degree);
rho_test1 = corre((Xtest * u1).^degree, (Ytest * v1).^degree);

%% Kernel Canonical Correlation Analysis

Kxtrain = centre_kernel((Xtrain * Xtrain').^degree);
Kytrain = centre_kernel((Ytrain * Ytrain').^degree);
Kxtest = centre_test_kernel((Xtest * Xtrain').^degree,(Xtrain * Xtrain').^degree);
Kytest = centre_test_kernel((Ytest * Ytrain').^degree,(Ytrain * Ytrain').^degree);

[alpha,beta,rho_kcc_train] = kcca_gep(Kxtrain,Kytrain,0.02,0.02,1);

u2 = preimage_kcca(Xtrain,alpha,degree);
v2 = preimage_kcca(Ytrain,beta,degree);

rho_train2 = corre((Xtrain * u2).^degree, (Ytrain * v2).^degree);
rho_test2 = corre((Xtest * u2).^degree, (Ytest * v2).^degree);

rho_kcc_test = corre(Kxtest * alpha, Kytest * beta);

%% SCCA-HSIC

[u3,v3] = scca_hsic(Xtrain,Ytrain,1,1);

rho_test3 = corre(Xtest * u3, Ytest * u3);
rho_train3 = corre(Xtrain * u3, Ytrain * v3);
            

%% F1 score and AUC

F1_gradKCCA = f1_score([u1;v1],[correct_u;correct_v]);
F1_KCCA = f1_score([u2;v2],[correct_u;correct_v]);
F1_SCCA_HSIC = f1_score([u3;v3],[correct_u;correct_v]);

AUC_gradKCCA = calc_auc([u1;v1]',[correct_u;correct_v]');
AUC_KCCA = calc_auc([u2;v2]',[correct_u;correct_v]');
AUC_SCCA_HSIC = calc_auc([u3;v3]',[correct_u;correct_v]');







