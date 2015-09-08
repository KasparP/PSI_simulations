function X = prox_DKL(Y,V,lambda)
% computes the proximal gradient of
% (1/lambda)*D_KL(Y,X) + (1/2)*norm(X-V,'fro').^2

A = (lambda*V - 1);
X = A + sqrt(A.^2 + 4*lambda*Y) / (2*lambda);