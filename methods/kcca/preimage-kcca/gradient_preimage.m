function [grad] = gradient_preimage(X,a,u,type)

switch type
    case 1 % linear kernel
        grad = 2 * u - 2 * X' * a;
        
    case 2 % quadratic kernel
        grad = 4 * (u' * u) * u - 4 * X' * (a .* (X * u));
    
    case 3 % cubic kernel
        grad = 2 * (u' * u).^2 * 3 * u - 6 * X' * (a .* (X * u).^2);
        
    case 4 % gaussian kernel
        sigma = 1;
        t0 = norm(u);
        t1 = a .* exp(2 * X * u - sqrt(sum(X.^2, 2)) * t0/(2*sigma^2));
        
        grad = -(4 * X' * t1 + 2/(2*sigma^2 * t0) * sum(t1) * u);
end
        
end

