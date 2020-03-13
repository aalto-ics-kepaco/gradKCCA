function g = gradf_poly(X,u,r,d,K)

% The gradient with respect to u (or v).
% input:        X - The training data, examples on rows, variables on
%               columns
%               u - current u
%               r - the constant in polynomial kernel
%               d - degree of the polynomial kernel
%               K - the projection of the other view


% output:       g - the gradient

t0 = r + X * u;
t1  = t0.^d;
t2 = 0.5;
t3 = -t2;
t4 = t1' * t1;
t5 = (K' * K)^t3;
t6 = t0.^(d-1);
t7 = (d * t5 * t4^(-(1+t2)))/2;
t8 = (t1 .* t6)'*X;

g = d * t5 * t4^t3 * (K .* t6)' * X - (t7 * K' * t1 * t8 + t7 * t1' * K * t8);



end