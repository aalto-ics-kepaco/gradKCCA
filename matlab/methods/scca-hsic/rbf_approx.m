function [phi, sigma] = rbf_approx(X, ind, varargin)

% The Nystrom approximated Gaussian RBF kernel.

% Input    
% X:                the data matrix to be approximated             
% ind:              indices of the Nystrom sample
% varargin          sigma (std) can given as third input variable,
%                   otherwise the median heuristic is applied

% Output
% phi:              the approximated kernel matrix 
% sigma             the standard deviation of the kernel
 
%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

n = ind; % nystrom sample
D_mn = sqrt(abs(sqdist(X',X(n,:)')));

if nargin == 2 % median heuristic
    sigma = median(D_mn(:));
else
    sigma = varargin{1};
end

K_mn = exp(- (D_mn.^2) ./ (2 * sigma.^2));

D_nn = sqrt(abs(sqdist(X(n,:)', X(n,:)')));
K_nn = exp(- (D_nn.^2) ./ (2 * sigma.^2));
K_nn = K_nn + eye(size(K_nn)) * 0.001;

phi = K_mn * inv(K_nn^(1/2));
    
     
end





