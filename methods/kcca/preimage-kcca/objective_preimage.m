function val = objective_preimage(X,alpha,u,type)

switch type
    
    case 1 % linear kernel
        val = alpha' * (X * X') * alpha - 2 * alpha' * (X * u) + (u' * u);
        
    case 2 % quadratic kernel
        val = alpha' * (X * X').^2 * alpha - 2 * alpha' * (X * u).^2 + (u' * u)^2;
        
    case 3 % cubic kernel
        val = alpha' * (X * X').^3 * alpha - 2 * alpha' * (X * u).^3 + (u' * u)^3;
        
    case 4 % gaussian kernel
        val = alpha' * rbf_kernel(X) * alpha - 2 * alpha' * rbf_kernel(X,u) + 1;
end


end

