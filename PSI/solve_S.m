function S = solve_S(@P,F,C,S0)
% Solve the generalized system of simultaneous Sylvester equations
% S + \sum_t P_t^T * P_t * S * F_t*F_t^T = C

tol = 1e-12;
maxit = 1000;

szS = size(C);
prodszS = prod(szS);
[Nproj,Nvox] = size(P);
[Nsrc,Nt] = size(F);

S = gmres(@Afun,reshape(C,prodszS,1),[],tol,maxit,[],[],reshape(S0,prodszS,1));
% S = pcg(@Afun,reshape(C,prodszS,1),tol,maxit,[],[],reshape(S0,prodszS,1));
S = reshape(S,szS);

function Chat = Afun(S)
	S = reshape(S,szS);
	Chat = S;
	for it = 1:T,
		Chat = Chat + P(:,:,it)'*(P(:,:,it)*S*F(:,it))*F(:,it)';
	end
	Chat = reshape(Chat,prodszS,1);
end
end