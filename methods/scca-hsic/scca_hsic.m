function [U,V,final_obj,tempobj,InterMediate] = scca_hsic(X,Y,Cx,Cy)

% The SCCA-HSIC implementation using the projected stochastic mini-batch
% gradient ascent.

% Input:
% X             n x dx data matrix
% Y             n x dy data matrix
%
% hyperparams structure with the following fields
% .M            number of components
% .normtypeX 	norm for X view 1 = l1 (default) and 2 = l2
% .normtypeY 	norm for Y view 1 = l1 and 2 = l2 (default)
% .Cx           the value of the norm constraint on view X
% .Cy           the value of the norm constraint on view Y
% .Rep          number of repetitions from random initializations
% .eps          convergence threshold
% .sigma1       the std of the rbf kernel, if empty = median heuristic
% .sigma2       the std of the rbf kernel, if empty = median heuristic
% .maxit        maximum iteration limit
% .flag         print iteration results, 1: yes, 2: only the converged
%               result

% Output:
% U             canonical coefficient vectors for X in the columns of U
% V             canonical coefficient vectors for Y in the columns of V

% InterMediate is a structure containing all intermediate results
% InterMediate(m,rep).u  contains all intermediate u for mth component
% InterMediate(m,rep).v  contains all intermediate v for mth component
% InterMEdiate(m,rep).obj contains intermediate objective values


%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J.
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion.
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------
%% Set up parameters

% M = hyperparams.M;
% normtypeX = hyperparams.normtypeX;
% normtypeY = hyperparams.normtypeY;
% Cx = hyperparams.Cx;
% Cy = hyperparams.Cy;
% Rep = hyperparams.Rep;
% eps = hyperparams.eps;
% sigma1 = hyperparams.sigma1;
% sigma2 = hyperparams.sigma2;
% maxit = hyperparams.maxit;
% flag = hyperparams.flag;

M = 1;
normtypeX = 1;
normtypeY = 1;
Rep = 8;
eps  = 1e-7;
sigma1 = [];
sigma2 = [];
maxit = 500;
flag = 0;

rng(5) % fix the random number generator

if ~exist('Rep', 'var') || isempty(Rep)
    Rep = 10;
end

if ~exist('eps', 'var') || isempty(eps)
    eps = 1e-6;
end

if ~exist('normtypeX', 'var') || isempty(normtypeX)
    normtypeX = 1; % default l1 norm for X
end

if ~exist('normtypeY', 'var') || isempty(normtypeY)
    normtypeY = 2; % default l2 norm for Y
end

if ~exist('Cx', 'var') || isempty(Cx)
    Cx = 1; % default regularization constant for X
end

if ~exist('Cy', 'var') || isempty(Cy)
    Cy = 1; % default regularization constant for Y
end


% partition into training and validation sets
[~,indices] = partition(size(X,1), 3);
train = indices ~= 1;
test = indices == 1;
Xtrain = X(train,:); Xtest = X(test,:);
Ytrain = Y(train,:); Ytest = Y(test,:);

Xm = Xtrain;
Ym = Ytrain;
dx = size(Xm,2);
dy = size(Ym,2);

if size(Xm,1) ~= size(Ym,1)
    printf('sizes of data matrices are not same');
end

InterMediate = [];
for m=1:M
    for rep=1:Rep
        %fprintf('Reps: #%d \n',rep);
        % intialization
        if normtypeX == 1
            umr = projL1(rand(dx,1),Cx);
        end
        if normtypeX == 2
            umr = projL2(rand(dx,1),Cx);
        end
        if normtypeY == 1
            vmr = projL1(rand(dy,1),Cy);
        end
        if normtypeY == 2
            vmr = projL2(rand(dy,1),Cy);
        end
        Xu = Xm * umr;
        Yv = Ym * vmr;
        
        % kernel for view x
        if isempty(sigma1)
            [Ku,au] = rbf_kernel1(Xu);
        else
            [Ku,au] = rbf_kernel1(Xu,sigma1);
        end
        % kernel fow view y
        if isempty(sigma2)
            [Kv,av] = rbf_kernel1(Yv);
        else
            [Kv,av] = rbf_kernel1(Yv,sigma2);
        end
        
        cKu = centre_kernel(Ku);
        cKv = centre_kernel(Kv);
        diff = 999999;
        ite = 0;
        
        while diff > eps && ite < maxit
            ite = ite + 1;
            obj_old = f_hsic(Ku,cKv);
            gradu = gradf_gauss_SGD(Ku,cKv,Xm,au,umr);
            
            %% line search for u
            gamma = norm(gradu,2); % initial step size
            chk = 1;
            while chk == 1
                if normtypeX == 1
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                if isempty(sigma1)
                    [Ku_new,au_new] = rbf_kernel1(Xm * umr_new);
                else
                    [Ku_new,au_new] = rbf_kernel1(Xm * umr_new,sigma1);
                end
                
                obj_new = f_hsic(Ku_new,cKv);
                
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    cKu = centre_kernel(Ku);
                    au = au_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-8
                        chk=0;
                        
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).u(:,ite) = umr;
            InterMediate(m,rep).obj(2*ite-1) = obj;
            %% line search end
            
            obj_old = obj;
            gradv = gradf_gauss_SGD(Kv,cKu,Ym,av,vmr);
            %% line search for v
            gamma = norm(gradv,2); % initial step size
            chk = 1;
            while chk == 1
                if normtypeY == 1
                    vmr_new  = projL1(vmr + gradv * gamma,Cy);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr + gradv * gamma,Cy);
                end
                
                if isempty(sigma2)
                    [Kv_new,av_new] = rbf_kernel1(Ym * vmr_new);
                else
                    [Kv_new,av_new] = rbf_kernel1(Ym * vmr_new, sigma2);
                end
                
                cKv_new = centre_kernel(Kv_new);
                obj_new = f_hsic(Ku,cKv_new);
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma <1e-8
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).v(:,ite) = vmr;
            InterMediate(m,rep).obj(2*ite) = obj;
            %% line search end
            %% check the value of test objective
            Kxtest = rbf_kernel1(Xtest * umr);
            Kytest = centre_kernel(rbf_kernel1(Ytest * vmr));
            test_obj = f_hsic(Kxtest,Kytest);
            
            %% compute the delta
            diff = abs(obj - obj_old) / abs(obj + obj_old);
            
            if flag == 1
                disp(['iter = ',num2str(ite),', objtr = ',num2str(obj),', diff = ', num2str(diff), ', test = ', num2str(test_obj)])
            end
        end
        InterMediate(m,rep).Result.u = umr;
        InterMediate(m,rep).Result.v = vmr;
        InterMediate(m,rep).Result.obj = obj;
        tempobj(rep) = obj;
        
        if flag == 2
            disp(['Rep ', num2str(rep), ', Objective = ',num2str(obj,2)])
        end
    end
    
    [~,id] = max(tempobj);
    U(:,m) = InterMediate(m,id).Result.u;
    V(:,m) = InterMediate(m,id).Result.v;
    final_obj(m,1) = max(tempobj);
    
    % deflated data
    %Xm = Xm - (U(:,m)*U(:,m)'*Xm')';
    %Ym = Ym - (V(:,m)*V(:,m)'*Ym')';
    
end
end












