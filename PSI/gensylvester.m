function X = gensylvester(A,B,C,X0)
% Solve the generalized system of simultaneous Sylvester equations
% \sum_i A_i * X * B_i^T = C

szX = size(C);
prodszX = prod(szX);
AXB = zeros(szX);
X = gmres(@Afun,reshape(C,prodszX,1),[],[],1000,[],[],reshape(X0,prodszX,1));
% X = pcg(@Afun,reshape(C,prodszX,1),1e-8,100,[],[],reshape(X0,prodszX,1));
X = reshape(X,szX);

function AXB = Afun(X)
	X = reshape(X,szX);
	AXB = zeros(szX);
	for k = 1:size(A,3),
		AXB = AXB + A(:,:,k)*X*B(:,:,k)';
	end
	AXB = reshape(AXB,prodszX,1);
end
end