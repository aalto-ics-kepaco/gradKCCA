function [K,sigma] = rbf_kernel1(X,varargin)

% The Gaussian RBF kernel.

% Input    
% X:                the data matrix to be approximated             
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

D = sqrt(abs(sqdist(X',X'))); % distance matrix

if nargin == 1 % median heuristic
    sigma = median(D(:));
else
    sigma = varargin{1};
end

K = exp(- (D.^2) ./ (2 * sigma.^2));

end 

