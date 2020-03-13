function y = projL2(x,r)

% The L2 norm projection
% input:        x - the vector to be projected
%               r - value of the norm constraint

% output:       y - the projected vector

y = x;
if norm(x) > 0.0001
    y = r * x / norm(x);
end
end