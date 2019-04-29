function y = projL2(x,r)

% Projection onto the L2 ball.

% Input
%       x: vector to be projected
%       r: L2 constraint
% Output
%       y: the projected vector

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

y = x;
if norm(x) > 0.0001
    y = r * x / norm(x);
end

end