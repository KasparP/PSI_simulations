function X = solve_M1_X_M2XM3(M1,M2,M3)
% Solve for X: M1 = X + M2*X*M3
% where
% X & M1 are N x M (rectangular)
% M2 is N x N, symmetric PSD
% M3 is M x M, symmetric PSD

[Q2,D2] = eig(M2); D2 = diag(D2);
[Q3,D3] = eig(M3); D3 = diag(D3);

M1bar = Q2'*M1*Q3;
Xbar =  M1bar ./ (1 + D2*D3');

X = Q2*Xbar*Q3';