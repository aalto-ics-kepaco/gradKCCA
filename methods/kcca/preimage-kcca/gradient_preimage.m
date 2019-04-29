function [grad] = gradient_preimage(X,a,u,type)

switch type
    case 1 % linear kernel
        grad = 2 * u - 2 * X' * a;
        
    case 2 % quadratic kernel
        grad = 4 * (u' * u) * u - 4 * X' * (a .* (X * u));
    
end
        
end

