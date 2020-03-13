
function [u_final] = preimage_kcca(X, alpha, type)

% This function finds the closest u that approximates the optimal KCCA
% solution.
%
% input: X - the training data
%        alpha - the optimal KCCA solution
%        type - 1 | Linear
%               2 | Quadratic

max_iters = 500;
iters = 0;
diff = 999999;
obs = [];

cur_u = projL2(rand(size(X,2),1), 1);

while iters < max_iters && diff > 1e-8
    
    obj_old = objective_preimage(X, alpha, cur_u, type);
    gradu = gradient_preimage(X, alpha, cur_u, type);
    gamma = norm(gradu)/2;
    
    ok = 1;
    while ok == 1 % backtracking line search
        u_new = projL2(cur_u - gradu * gamma, 1);
        cur_obj = objective_preimage(X, alpha ,u_new ,type);
        
        if cur_obj < obj_old % gradient descent update
            ok = 0;
            obj = cur_obj;
            cur_u = u_new;
        else
            gamma = gamma/2;
            if gamma < 1e-11
                ok = 0;
            end
        end
        
    end
    obj = cur_obj;
    diff = abs(obj - obj_old) / abs(obj + obj_old);
    iters = iters + 1;
    %disp(['ite #',num2str(iters), ' obj ',num2str(obj), ' diff ', num2str(diff)])
end

u_final = u_new;



