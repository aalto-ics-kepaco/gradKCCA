function val = objective_preimage(X,alpha,u,type)

switch type
    
    case 1 % linear kernel
        val = alpha' * (X * X') * alpha - 2 * alpha' * (X * u) + (u' * u);
        
    case 2 % quadratic kernel
        val = alpha' * (X * X').^2 * alpha - 2 * alpha' * (X * u).^2 + (u' * u)^2;
        
end


end

