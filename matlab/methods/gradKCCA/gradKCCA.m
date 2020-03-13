function [U,V] = gradKCCA(X,Y,M,hyperparams)

% gradKCCA finds the canonical coefficient vectors u and v that are a
% solution to the optimization problem (6) in the manuscript "Large-Scale
% Sparse Kernel Canonical Correlation Analysis".

% input:        X - the n x p training data from view X
%               Y - the n x q training data from view Y
%               M - number of components
%               hyperparams structure:
%               .normtypeX - norm constraint on u
%               .normtypeY - norm constraint on v
%               .Cx - value of the norm constraint on u
%               .Cy - value of the norm constraint on v
%               .Rep - number of random initializations of u and v
%               .eps - stopping criterion
%               .degree1 - the degree of the polynomial kernel on Xu
%               .degree2 - the degree of the polynomial kernel on Yv

% output:       U - the u vectors
%               V - the v vectors


normtypeX = hyperparams.normtypeX;
normtypeY = hyperparams.normtypeY;
Cx = hyperparams.Cx;
Cy = hyperparams.Cy;
Rep = hyperparams.Rep;
eps = hyperparams.eps;
degree1 = hyperparams.degree1;
degree2 = hyperparams.degree2;

Xm = X;
Ym = Y;
dx = size(Xm,2);
dy = size(Ym,2);
maxit = 600;
r = 0;
InterMediate=[];
for m=1:M
    for rep=1:Rep
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
        
        Ku = polyK(Xm, umr, degree1);
        Kv = polyK(Ym,vmr,degree2);
        cKu = Ku;
        cKv = Kv;
        diff = 999999;
        ite = 0;
        while diff > eps && ite < maxit
            ite = ite + 1;
            obj_old = f_gkcca(Ku,cKv);
            % GRADIENT
            gradu = gradf_poly(Xm, umr, r, degree1, cKv)';
            
            % LINE SEARCH---------------%
            gamma = norm(gradu,2);
            chk = 1;
            while chk == 1
                if normtypeX == 1
                    umr_new  = projL1(umr + gradu * gamma, Cx);
                end
                if normtypeX == 2
                    umr_new  = projL2(umr + gradu * gamma, Cx);
                end
                
                [Ku_new,au_new] = polyK(Xm, umr_new, degree1);
                obj_new = f_gkcca(Ku_new,cKv);
                
                if obj_new > obj_old + 1e-4 * abs(obj_old)
                    chk = 0;
                    umr = umr_new;
                    Ku = Ku_new;
                    cKu = Ku;
                    au = au_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-13
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).u(:,ite) = umr;
            InterMediate(m,rep).obj(2*ite-1) = obj;
            %-------------------------------------%
            
            obj_old = obj;
            gradv = gradf_poly(Ym,vmr,r,degree2,cKu)';
            %line search for v
            gamma = norm(gradv,2);
            chk = 1;
            while chk==1
                if normtypeY == 1
                    vmr_new  = projL1(vmr + gradv * gamma,Cy);
                end
                if normtypeY == 2
                    vmr_new  = projL2(vmr + gradv * gamma,Cy);
                end
                [Kv_new,av_new] = polyK(Ym, vmr_new,degree2);                
                cKv_new = Kv_new;                
                obj_new = f_gkcca(Ku,cKv_new);
                if obj_new > obj_old + 1e-4*abs(obj_old)
                    chk = 0;
                    vmr = vmr_new;
                    Kv = Kv_new;
                    cKv = cKv_new;
                    av = av_new;
                    obj = obj_new;
                else
                    gamma = gamma/2;
                    if gamma < 1e-13
                        chk = 0;
                    end
                end
            end
            obj = obj_new;
            InterMediate(m,rep).v(:,ite) = vmr;
            InterMediate(m,rep).obj(2*ite) = obj;            
            %line search end
            diff = abs(obj-obj_old)/abs(obj+obj_old);
            
            %disp(['iter = ',num2str(ite),', objtr = ',num2str(obj), ', diff = ', num2str(diff)])
        end
        InterMediate(m,rep).Result.u = umr;
        InterMediate(m,rep).Result.v = vmr;
        InterMediate(m,rep).Result.obj = obj;
        tempobj(rep) = obj;
    end
    
    [~,id] = max(tempobj);
    U(:,m) = InterMediate(m,id).Result.u;
    V(:,m) = InterMediate(m,id).Result.v;
    final_obj(m,1) = max(tempobj);
    
    % deflated data
    %Xm = Xm - (U(:,m)*U(:,m)'*Xm')';
    %Ym = Ym - (V(:,m)*V(:,m)'*Ym')';
    
end













