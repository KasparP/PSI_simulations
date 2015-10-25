function S = solve_S_Pfun(P,F,C,S0,maxit)
% Solve the generalized system of simultaneous Sylvester equations
% S + \sum_t P_t^T * P_t * S * F_t*F_t^T = C

if ~exist('maxit','var') || isempty(maxit),
	maxit = 1;
end
tol = 1e-12;

szS = size(C);
prodszS = prod(szS);
[Nproj,Nvox] = size(P(1));
[Nsrc,T] = size(F);

% S = gmres(@Afun,reshape(C,prodszS,1),[],tol,maxit,[],[],reshape(S0,prodszS,1));
S = pcg(@Afun,reshape(C,prodszS,1),tol,maxit,[],[],reshape(S0,prodszS,1));
S = reshape(S,szS);

function Chat = Afun(S)
	S = reshape(S,szS);
	Chat = S;
	for it = 1:T,
		Chat = Chat + P(it)'*(P(it)*(S*F(:,it)))*F(:,it)';
	end
	Chat = reshape(Chat,prodszS,1);
end
end