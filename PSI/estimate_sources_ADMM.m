function [Su,Fu,Sk,Fk] = estimate_sources_ADMM(Y,P,Mk,Su,Fu,Sk,Fk, ...
										lambdaFu_nuc,lambdaFu_TF, ...
										lambdaFk_nuc,lambdaFk_TF)

[Np T] = size(Y);
[Np Nv] = size(P);
[Nsu T] = size(Fu);
[Nsk T] = size(Fk);

Fu_nuc = zeros(size(Fu));
Fu_nn = zeros(size(Fu));
Fu_TF = zeros(size(Fu));
Fu_TF_z = zeros(size(Fu));
Fu_TF_u = zeros(size(Fu));
Fk_nuc = zeros(size(Fk));
Fk_nn = zeros(size(Fk));
Fk_TF = zeros(size(Fk));
Fk_TF_z = zeros(size(Fk));
Fk_TF_u = zeros(size(Fk));

U_X = 0*ones(Np,T);
U_Fu_nuc = 0*ones(Nsu,T);
U_Fu_TF = 0*ones(Nsu,T);
U_Fu_nn = 0*ones(Nsu,T);
U_Fk_nuc = 0*ones(Nsk,T);
U_Fk_TF = 0*ones(Nsk,T);
U_Fk_nn = 0*ones(Nsk,T);
U_Su_nn = 0*ones(Nv,Nsu);
U_Sk_nn = 0*ones(Nv,Nsk);

for it = 1:T,
	PSu(:,:,it) = P(:,:,it)*Su;
	PSk(:,:,it) = P(:,:,it)*Sk;
	PSFu(:,it) = PSu(:,:,it)*Fu(:,it);
	PSFk(:,it) = PSk(:,:,it)*Fk(:,it);
	Xhat(:,it) = PSFu + PSFk;
end

X = prox_DKL(Y,U_X-Xhat,1/rho);

Fu_nuc = prox_matrix(Fu-U_Fu_nuc,lambdaFu_nuc/rho,@prox_l1);
Fk_nuc = prox_matrix(Fk-U_Fk_nuc,lambdaFk_nuc/rho,@prox_l1);

[Fu_TF, Fu_TF_z, Fu_TF_u] = prox_matrix_L1(D,Fu-U_Fu_TF,lambdaFu_TF,rho,1,10,Fu_TF,Fu_TF_z,Fu_TF_u);
[Fk_TF, Fk_TF_z, Fk_TF_u] = prox_matrix_L1(D,Fk-U_Fk_TF,lambdaFk_TF,rho,1,10,Fk_TF,Fk_TF_z,Fk_TF_u);

Fu_nn = max(0,Fu-U_Fu_nn);
Fk_nn = max(0,Fk-U_Fk_nn);

Su_nn = max(0,Su-U_Su_nn);
Sk_nn = Mk*max(0,Sk-U_Sk_nn);

U_X = U_X+Xhat-X;

U_Fu_nuc = U_Fu_nuc+Fu_nuc-Fu;
U_Fk_nuc = U_Fk_nuc+Fk_nuc-Fk;

U_Fu_TF = U_Fu_TF+Fu_TF-Fu;
U_Fk_TF = U_Fk_TF+Fk_TF-Fk;

U_Fu_nn = U_Fu_nn+Fu_nn-Fu;
U_Fk_nn = U_Fk_nn+Fk_nn-Fk;

U_Su_nn = U_Su_nn+Su_nn-Su;
U_Sk_nn = U_Sk_nn+Sk_nn-Sk;

U_X_PSFk = U_X+X-PSFk;
U_X_PSFu = U_X+X-PSFu;

for it = 1:T,
	Fu(:,t) = (PSu(:,:,it)'*PSu(:,:,it)+3*eye(Nsu)) \ (PSu(:,:,it))'*(U_X_PSFk(:,:,it) - (U_Fu_nuc(:,t)+U_Fu_TF(:,t)+U_Fu_nuc(:,t)) + (Fu_nuc(:,t)+Fu_TF(:,t)+Fu_nn(:,t)));
	Fk(:,t) = (PSk(:,:,it)'*PSk(:,:,it)+3*eye(Nsk)) \ (PSk(:,:,it))'*(U_X_PSFu(:,:,it) - (U_Fk_nuc(:,t)+U_Fk_TF(:,t)+U_Fk_nuc(:,t)) + (Fk_nuc(:,t)+Fk_TF(:,t)+Fk_nn(:,t)));
end

C = U_Su_nn + Su_nn;
for it = 1:T,
	C = C + P(:,:,it)'*U_X_PSFk(:,it)*Fu(:,it)';
end
Su = solve_S(P,Fu,C,Su);

C = U_Sk_nn + Sk_nn;
for it = 1:T,
	C = C + P(:,:,it)'*U_X_PSFu(:,it)*Fk(:,it)';
end
Su = solve_S(P,Fu,C,Su);
Sk = solve_S(P,Fk,C,Sk);




%%
	function [l,l_aug] = loss;
		l = 0;
		l_aug = 0;
	end

	function Pmotioncorrect = Pit(it)
		Pmotioncorrect = P;
	end


end
