function [X,Y] = generate_algebraic_relations(n,p,q,numx,type)

% Function to generate simulated algebraic relations.
% input:        n - sample size
%               p - number of variables in X view
%               q - number of variables in Y view
%               numx - number of related variables, 2 means a two-to-two
%               relation
%               type - (1) linear (2) quadratic (3) cubic (4) exponential
%               (5) logarithmic

% output:       X - the simulated data X
%               Y - the simulated data Y

a = -1; b = 1;
X = a + (b-a).*rand(n,p);
Y = a + (b-a).*rand(n,q);

xvar = zeros(size(X,1),1);
yvar = zeros(size(X,1),1);
if numx >= 2
    for k = 2:numx
        xvar = xvar + X(:,k);
        yvar = yvar + Y(:,k);
    end
end

switch type
    case 1
        Y(:,1) = (X(:,1) + xvar) - yvar + normrnd(0,0.05,[n,1]);
    case 2
        Y(:,1) = (X(:,1) + xvar).^2 - yvar + normrnd(0,0.05,[n,1]);
    case 3
        Y(:,1) = (X(:,1) + xvar).^3 - yvar + normrnd(0,0.05,[n,1]);
    case 4
        Y(:,1) = exp(X(:,1) + xvar) - yvar + normrnd(0,0.05,[n,1]);
end






end
