function [Su,Fu,Sk,Fk] = estimate_sources_ADMM(Y,P,Su,Fu,Sk,Fk, ...
										lambdaFu_nuc,lambdaFu_TF, ...
										lambdaFk_nuc,lambdaFk_TF)

[Np T] = size(Y);
[Np Nv] = size(P);
[Nsu T] = size(Fu);
[Nsk T] = size(Fk);

U_X = 0*ones(Np,T);
U_Fu_nuc = 0*ones(Nsu,T);
U_Fu_TF = 0*ones(Nsu,T);
U_Fu_nn = 0*ones(Nsu,T);
U_Fk_nuc = 0*ones(Nsk,T);
U_Fk_TF = 0*ones(Nsk,T);
U_Fk_nn = 0*ones(Nsk,T);
U_Su_nn = 0*ones(Nv,Nsu);
U_Sk_nn = 0*ones(Nv,Nsk);

PSu = P*Su;
PSk = P*Sk;
PSFu = PSu*Fu;
PSFk = PSk*Fk;
Xhat = PSFu + PSFk;

X = prox_DKL(Y,U_X-Xhat,1/rho);

Fu_nuc = prox_matrix(U_Fu_nuc-Fu,lambdaFu_nuc/rho,@prox_l1);
Fk_nuc = prox_matrix(U_Fk_nuc-Fk,lambdaFk_nuc/rho,@prox_l1);

Fu_TF = prox_matrix_L1(D,U_Fu_TF-Fu,lambdaFu_TF,rho,1,10);
Fk_TF = prox_matrix_L1(D,U_Fk_TF-Fk,lambdaFk_TF,rho,1,10);

Fu_nn = max(0,U_Fu_nn-Fu);
Fk_nn = max(0,U_Fk_nn-Fk);

Su_nn = max(0,U_Su_nn-Su);
Sk_nn = max(0,U_Sk_nn-Sk);

U_X = U_X+Xhat-X;

U_Fu_nuc = U_Fu_nuc+Fu_nuc-Fu;
U_Fk_nuc = U_Fk_nuc+Fk_nuc-Fk;

U_Fu_TF = U_Fu_TF+Fu_TF-Fu;
U_Fk_TF = U_Fk_TF+Fk_TF-Fk;

U_Fu_nn = U_Fu_nn+Fu_nn-Fu;
U_Fk_nn = U_Fk_nn+Fk_nn-Fk;

U_Su_nn = U_Su_nn+Su_nn-Su;
U_Sk_nn = U_Sk_nn+Sk_nn-Sk;

Fu = (PSu'*PSu+3*eye(Nsu)) \ (PSu)'*(U_X+X-PSFk - (U_Fu_nuc+U_Fu_TF+U_Fu_nuc) + (Fu_nuc+Fu_TF+Fu_nn));
Fk = (PSk'*PSk+3*eye(Nsk)) \ (PSk)'*(U_X+X-PSFu - (U_Fk_nuc+U_Fk_TF+U_Fk_nuc) + (Fk_nuc+Fk_TF+Fk_nn));

M1 = P'*(U_X+X-PSFk)*Fu' + U_Su_nn + Su_nn;
M2 = P'*P;
M3 = Fu*Fu';
Su = solve_A_X_B2XC(M1,M2,M3);

M1 = P'*(U_X+X-PSFu)*Fk' + U_Sk_nn + Sk_nn;
M3 = Fk*Fk';
Sk = solve_A_X_B2XC(M1,M2,M3);

	function [l,l_aug] = loss;
		l = 0;
		l_aug = 0;
	end

end
