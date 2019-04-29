function [X,Y] = generate_trigonometric_relations(n,p,q,type)

% Function to generate simulated trigonometric relations.
% input:        n - sample size
%               p - number of variables in X view
%               q - number of variables in Y view
%               type - the numbers 1-3 correspond to the relations
%               described in the manuscript "Large-Scale Sparse Kernel
%               Canonical Correlation Analysis", Section 5.1

% output:       X - the simulated data X
%               Y - the simulated data Y
        
        
rng('shuffle')
a = -2*pi; b = 2*pi;
X = a + (b-a).*rand(n,p);
Y = a + (b-a).*rand(n,q);
   
switch type
    case 1 
        X(:,1) = sin(X(:,2)/-2) + normrnd(0,0.05,[n,1]);
        Y(:,1) = cos(X(:,2)/3) + normrnd(0,0.05,[n,1]);
        Y(:,2) = cos(X(:,2)/4) + normrnd(0,0.05,[n,1]);
        
    case 2 
        X(:,1) = sin(X(:,2)/-2) + normrnd(0,0.05,[n,1]);
        Y(:,1) = cos(X(:,2)/-4) + normrnd(0,0.05,[n,1]);
        Y(:,2) = cos(X(:,2)/-6) + normrnd(0,0.05,[n,1]);
        
    case 3 
        X(:,1) = sin(X(:,2)/-0.5) + normrnd(0,0.05,[n,1]);
        Y(:,1) = cos(X(:,2)/2) + normrnd(0,0.05,[n,1]);
        Y(:,2) = cos(X(:,2)/-2) + normrnd(0,0.05,[n,1]);



end





end
