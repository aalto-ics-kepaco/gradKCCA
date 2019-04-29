function [U,V,final_obj,tempobj,InterMediate] = scca_hsic_nystrom(X,Y,hyperparams)

% The Nystrom approximated SCCA-HSIC implementation using the projected
% stochastic mini-batch gradient ascent.

% Input:
% X             n x dx data matrix
% Y             n x dy data matrix
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
% .flag         print results

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

M = hyperparams.M;
normtypeX = hyperparams.normtypeX;
normtypeY = hyperparams.normtypeY;
proportion = hyperparams.proportion;
Cx = hyperparams.Cx;
Cy = hyperparams.Cy;
Rep = hyperparams.Rep;
eps = hyperparams.eps;
sigma1 = hyperparams.sigma1;
sigma2 = hyperparams.sigma2;
maxit = hyperparams.maxit;
flag = hyperparams.flag;

Xm = X;
Ym = Y;
N = size(Xm,1);
dx = size(Xm,2);
dy = size(Ym,2);
Nnym = ceil(proportion * N);

InterMediate = [];
for m = 1:M % for every component
    for rep = 1:Rep % rep times
        %fprintf('Reps: #%d \n',rep);
        
        % initialize the u and v
        if normtypeX==1
            umr = projL1(rand(dx,1),Cx);
        end
        if normtypeX==2
            umr = projL2(rand(dx,1),Cx);
        end
        if normtypeY==1
            vmr = projL1(rand(dy,1),Cy);
        end
        if normtypeY==2
            vmr = projL2(rand(dy,1),Cy);
        end
        
        % random sampling
        ind = randperm(N, Nnym);
        
        % compute the approximated kernel
        if sigma1 > 0
            [phiu, au] = rbf_approx(Xm * umr, ind, sigma1);
        else
            [phiu, au] = rbf_approx(Xm * umr, ind);
        end
        Ku = phiu' * phiu;
        
        if sigma2 > 0
            [phiv, av] = rbf_approx(Ym * vmr, ind, sigma2);
        else
            [phiv, av] = rbf_approx(Ym * vmr, ind);
        end
        Kv = phiv' * phiv;
        
        % centre the kernels
        [phicu] = centre_nystrom_kernel(phiu);
        cKu = phicu' * phicu;
        [phicv] = centre_nystrom_kernel(phiv);
        cKv = phicv' * phicv;
        
        diff = 999999;
        ite = 0;
        obj_old = f_nystrom(phicu,phicv);
        
        while diff > eps && ite < maxit  % stopping conditions
            ite = ite + 1;
            obj = obj_old;
            
            % GRADIENT WRT U
            gradu = gradf_gauss_SGD(Ku ,cKv ,Xm(ind,:), au ,umr);
            
            % LINE SEARCH FOR U
            gamma = norm(gradu,2);
            chk = 1;
            while chk == 1
                if normtypeX == 1
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                if sigma1 > 0
                    [phiu_new, au_new] = rbf_approx(Xm * umr_new, ind, sigma1);
                else
                    [phiu_new, au_new] = rbf_approx(Xm * umr_new, ind);
                end
                Ku_new = phiu_new' * phiu_new;
                
                phicu_new = centre_nystrom_kernel(phiu_new);
                cKu_new = phicu_new' * phicu_new;
                obj_new = f_nystrom(phicu_new, phicv);
                
                if obj_new > obj_old + 1e-4 * abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    au = au_new;
                    obj = obj_new;
                    cKu = cKu_new;
                    phicu = phicu_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-8
                        chk = 0;
                    end
                end
            end
            
            obj = obj_new;
            InterMediate(m,rep).u(:,ite) = umr;
            InterMediate(m,rep).obj(2*ite-1) = obj;
            % LINE SEARCH END
            
            obj_old = obj;
            % GRADIENT WRT V
            gradv = gradf_gauss_SGD(Kv,cKu,Ym(ind,:),av,vmr);
            
            % LINE SEARCH FOR V
            gamma = norm(gradv, 2);
            chk = 1;
            while chk == 1
                if normtypeY == 1
                    vmr_new  = projL1(vmr + gradv * gamma, Cx);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr + gradv * gamma,Cy);
                end
                
                if sigma2 > 0
                    [phiv_new, av_new] = rbf_approx(Ym * vmr_new, ind, sigma2);
                else
                    [phiv_new, av_new] = rbf_approx(Ym * vmr_new, ind);
                end
                Kv_new = phiv_new' * phiv_new;
                
                [phicv_new] = centre_nystrom_kernel(phiv_new);
                cKv_new = phicv_new' * phicv_new;
                obj_new = f_nystrom(phicu,phicv_new);
                
                if obj_new > obj_old + 1e-4 * abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    phicv = phicv_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-8
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).v(:,ite) = vmr;
            InterMediate(m,rep).obj(2*ite) = obj;
            % LINE SEARCH END

            
            diff = abs(obj - obj_old) / abs(obj + obj_old);
            
            if flag == 1
                disp(['iter = ',num2str(ite),', objtr = ',num2str(obj),...
                    ', diff = ', num2str(diff)])
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












