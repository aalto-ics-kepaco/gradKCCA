function obj = f_hsic(Kx,cKy)

% The HSIC objective.

% Input    
% Kx:   uncentred kernel matrix of view x
% cKy:  centred kernel matrix of view y

% Output
% obj:  value of HSIC

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

N = size(Kx,1);
obj = trace(Kx*cKy)/(N-1)^2;


end