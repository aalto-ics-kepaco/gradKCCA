function obj = f_gkcca(Kx,cKy)

% The objective function, e.g. Equation (5) in the manuscript.
% input:        Kx - the projection on X view
%               cKy - the projection on Y view

% output:       obj - the value of the correlation

obj = Kx' * cKy / (sqrt(Kx' * Kx) * sqrt(cKy' * cKy)) ;

end