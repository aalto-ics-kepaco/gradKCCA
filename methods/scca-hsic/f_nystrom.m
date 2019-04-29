function [hsic_nystrom] = f_nystrom(phix, phiy)

% The HSIC objective.

% Input    
% phix:             centred approximated matrix of view x
% phiy:             centred approximated matrix of view y

% Output
% hsic_nystrom:     value of HSIC

%--------------------------------------------------------------------------
% Uurtio, V., Bhadra, S., Rousu, J. 
% Sparse Non-Linear CCA through Hilbert-Schmidt Independence Criterion. 
% IEEE International Conference on Data Mining (ICDM 2018)
%--------------------------------------------------------------------------

hsic_nystrom = norm(1/size(phix,1) * phix' * phiy,'fro')^2;

end

