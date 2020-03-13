function Ktt_n = centre_test_kernel(Ktest, Ktrain)

% input
%       Ktest - test kernel matrix, e.g. Xtest * Xtrain'
%       Ktrain - training kernel matrix, e.g. Xtrain * Xtrain' (not
%       centered)
% output
%       Kc - centered test kernel matrix

ntr = size(Ktrain,1);
ntt = size(Ktest,1);
unit = ones(ntr, ntr)/ntr;
unit_test = ones(ntt,ntr)/ntr;
Ktt_n = Ktest - unit_test * Ktrain - Ktest * unit + unit_test * Ktrain * unit; 

end