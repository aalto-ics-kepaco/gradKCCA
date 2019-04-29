function [K,degree] = polyK(X,u,degree)

% The gradient with respect to u (or v).
% input:        X - The training data, examples on rows, variables on
%               columns
%               u - current u
%               degree - degree of the polynomial kernel

% output:       K - the kernelized projection

K = X * u;
r = 0;
if degree > 0
    K = (K + r).^degree;
end

end