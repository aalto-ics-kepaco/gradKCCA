function [alpha,beta,cc,za,zb] = kcca_gep(Ka,Kb,c1,c2,rels)
% This function solves kernel canonical correlation analysis through the
% generalised eigenvalue problem.

% Uurtio et al. A Tutorial on Canonical Correlation Methods. 2017.
%--------------------------------------------------------------------------
% Input 
%       Ka: the Gram matrix of the 1st view (view a)
%       Kb: the Gram matrix of the 2nd view (view b)
%       c1: the regularisation parameter for the 1st view
%       c2: the regularisation parameter for the 2nd view
%       rels: number of relations of interest
% Output
%       za: the image of the position wa (a.k.a. canonical variate)
%       zb: the image of the position wb (a.k.a. canonical variate)
%       alpha: the position in the data space of view a
%       beta: the position in the data space of view b
%       cc: the canonical correlation (cosine of the enclosing angle)
%--------------------------------------------------------------------------
% © 19/01/2017 Viivi Uurtio, Aalto University
% viivi.uurtio@aalto.fi
%
% This code is for academic purposes only.
% Commercial use is not allowed.
%--------------------------------------------------------------------------

n = size(Ka,1);

% regularise
Kar = Ka + c1 * eye(size(Ka));
Kbr = Kb + c2 * eye(size(Kb));

A = [zeros(n) Ka*Kb;
    Kb*Ka zeros(n)];
B = [Kar^2 zeros(n);
    zeros(n) Kbr^2];

[Vg,Dg] = eig(A,B);
rhosg = diag(Dg); % canonical correlations
[ev, ind] = sort(rhosg,'descend'); % sort in descending order
vecsg = Vg(:,ind); 

alpha = vecsg(1:n,1:rels);
beta = vecsg(n+1:end,1:rels);

za = normc(Ka * alpha); 
zb = normc(Kb * beta); 

for i = 1:rels
    cc(i) = za(:,i)'*zb(:,i);
end
[cc, ix] = sort(cc,'descend');
za = za(:,ix); zb = zb(:,ix);


end

