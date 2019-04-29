function [phic] = centre_nystrom_kernel(phi)

% This function centres the approximated kernel matrix.

% Input
%       phi: uncentred, approximated data matrix
% Output
%       phic: centred, approximated data matrix

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

phic = (eye(size(phi,1)) - 1 / size(phi,1) * ones(size(phi,1))) * phi;


end

