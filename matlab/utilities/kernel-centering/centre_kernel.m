function K = centre_kernel(K)

% This function centres the kernel matrix.

% Input
%       K: uncentred kernel matrix
% Output
%       phic: centred kernel matrix

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

K = K + mean(K(:)) - ...
    (repmat(mean(K,1),[size(K,1),1])+repmat(mean(K,2),[1,size(K,2)]));

end