function R = reconstruct_imaging(obs,opts,nIter,groundtruth)
%Estimates the intensity pattern of the sample given the observations

Nframes = 10;

rho = 1e-3
lambdaFk_nuc = 1e-10;
lambdaFu_nuc = 1e-10;
lambdaFk_TF = 1e-10;
lambdaFu_TF = 1e-10;

% obs.data_in = obs.data_in(:,1:T);
mask = groundtruth.seg.bw(:)>0;
Nvoxk = sum(mask);
obs.data_in = opts.P(mask,:)'*groundtruth.seg.seg(mask,:)*groundtruth.activity;

[Nvox,Np] = size(opts.P);
[Np,T] = size(obs.data_in);

Nsk = size(groundtruth.activity,1);
Nsu = size(groundtruth.unsuspected.pos,2);

Sk0 = groundtruth.seg.seg(mask,:);
Fk0 = groundtruth.activity(:,1:T);

%%
fprintf('Initializing variables...'),tic

Y = obs.data_in;
P0 = opts.P';

X = zeros(Np,T);
Xhat = zeros(Np,T);
Xhat_nn = zeros(Np,T);

Sk = 1e-3*rand(Nvoxk,Nsk);%+Sk0;
Fk = 5e-1*rand(Nsk,T);%+Fk0;

Su = 1e-3*rand(Nvox,Nsu);
Fu = 5e-2*rand(Nsu,T);

[Np Nvox] = size(P0);
[Nsu T] = size(Fu);
[Nsk T] = size(Fk);

Fu_nuc = Fu;%zeros(size(Fu));
Fu_nn = Fu;%zeros(size(Fu));
Fu_TF = Fu;%zeros(size(Fu));
Fu_TF_z = Fu;%zeros(size(Fu));
Fu_TF_u = Fu;%zeros(size(Fu));
Fk_nuc = Fk;%zeros(size(Fk));
Fk_nn = Fk;%zeros(size(Fk));
Fk_TF = Fk;%zeros(size(Fk));
Fk_TF_z = Fk;%zeros(size(Fk));
Fk_TF_u = Fk;%zeros(size(Fk));

U_X = 0*ones(Np,T);
U_Fu_nuc = 0*ones(size(Fu));
U_Fu_TF = 0*ones(size(Fu));
U_Fu_nn = 0*ones(size(Fu));
U_Fk_nuc = 0*ones(size(Fk));
U_Fk_TF = 0*ones(size(Fk));
U_Fk_nn = 0*ones(size(Fk));
U_Su_nn = 0*ones(size(Su));
U_Sk_nn = 0*ones(size(Sk));

fprintf(' done. '), toc


	tidx = randsample(T,Nframes)';
for iter = 1:nIter,

	% subsample in time
	tidx = randsample(T,Nframes)';
	tidx2 = tidx;
	if rem(iter,50)==1,
		tidx2=1:T;
	end

	fprintf('Pre-computing P*[Sk*Fk+Su*Fu]... '), tic
	% b=0;
	for it = tidx2,
		PSFu(:,it) = Pu(it)*(Su*Fu(:,it));
		PSFk(:,it) = Pk(it)*(Sk*Fk(:,it));
		Xhat_nn(:,it) = Pu(it)*(max(Su,0)*max(Fu(:,it),0)) + Pk(it)*(max(Sk,0)*max(Fk(:,it),0));
		% fprintf([repmat('\b',1,b)]); b=fprintf('%d',it);
	end
	Xhat(:,tidx2) = PSFu(:,tidx2) + PSFk(:,tidx2);
	fprintf('done. '), toc

	X(:,tidx2) = prox_DKL(Y(:,tidx2),Xhat(:,tidx2)-U_X(:,tidx2),rho);
	U_X_PSFk(:,tidx2) = U_X(:,tidx2)+X(:,tidx2)-PSFk(:,tidx2);
	U_X_PSFu(:,tidx2) = U_X(:,tidx2)+X(:,tidx2)-PSFu(:,tidx2);
	% U_X_PSFk = X-PSFk-U_X;
	% U_X_PSFu = X-PSFu-U_X;

	Fu_nuc = prox_matrix(Fu-U_Fu_nuc,lambdaFu_nuc/rho,@prox_l1);
	Fk_nuc = prox_matrix(Fk-U_Fk_nuc,lambdaFk_nuc/rho,@prox_l1);

	% [Fu_TF, Fu_TF_z, Fu_TF_u] = prox_matrix_L1(D,Fu-U_Fu_TF,lambdaFu_TF,rho,1,10,Fu_TF,Fu_TF_z,Fu_TF_u);
	% [Fk_TF, Fk_TF_z, Fk_TF_u] = prox_matrix_L1(D,Fk-U_Fk_TF,lambdaFk_TF,rho,1,10,Fk_TF,Fk_TF_z,Fk_TF_u);

	Fu_nn(:,tidx2) = max(0,Fu(:,tidx2)-U_Fu_nn(:,tidx2));
	Fk_nn(:,tidx2) = max(0,Fk(:,tidx2)-U_Fk_nn(:,tidx2));

	% rectify, and normalize to 1
	Su_nn = max(0,Su-U_Su_nn);
	Su_nn = bsxfun(@times,Su_nn,1./sum(Su_nn,1));
	Sk_nn = max(0,Sk-U_Sk_nn);
	Sk_nn = bsxfun(@times,Sk_nn,1./sum(Sk_nn,1));

	%% Update consensus variables
	fprintf('Estimating F, t='), tic
	b=0;
	for it = tidx2,
		PSu = Pu(it)*Su;
		PSk = Pk(it)*Sk;
		% Fu(:,it) = (PSu'*PSu+3*eye(Nsu)) \ (PSu'*U_X_PSFk(:,it) - (U_Fu_nuc(:,it)+U_Fu_TF(:,it)+U_Fu_nuc(:,it)) + (Fu_nuc(:,it)+Fu_TF(:,it)+Fu_nn(:,it)));
		% Fk(:,it) = (PSk'*PSk+3*eye(Nsk)) \ (PSk'*U_X_PSFu(:,it) - (U_Fk_nuc(:,it)+U_Fk_TF(:,it)+U_Fk_nuc(:,it)) + (Fk_nuc(:,it)+Fk_TF(:,it)+Fk_nn(:,it)));
		Fu(:,it) = (PSu'*PSu+2*eye(Nsu)) \ (PSu'*U_X_PSFk(:,it) + (U_Fu_nuc(:,it)+U_Fu_nuc(:,it)) + (Fu_nuc(:,it)+Fu_nn(:,it)));
		Fk(:,it) = (PSk'*PSk+2*eye(Nsk)) \ (PSk'*U_X_PSFu(:,it) + (U_Fk_nuc(:,it)+U_Fk_nuc(:,it)) + (Fk_nuc(:,it)+Fk_nn(:,it)));
		fprintf([repmat('\b',1,b)]); b=fprintf('%d',it);
	end
	fprintf('. Done. '), toc

	fprintf('Estimating S... \n'), tic
	Cu = U_Su_nn + Su_nn;
	Tidx = 1:T;%randsample(T,10);
	for it = tidx;%1:length(Tidx),
		Cu = Cu + Pu(it)'*U_X_PSFk(:,it)*Fu(:,it)';
	end
	Su = solve_S_Pfun(@Pu,Fu,Cu,Su,tidx);

	Ck = U_Sk_nn + Sk_nn;
	for it = tidx;%1:length(Tidx),
		Ck = Ck + Pk(it)'*U_X_PSFu(:,it)*Fk(:,it)';
	end
	Sk = solve_S_Pfun(@Pk,Fk,Ck,Sk,tidx);
	fprintf('Done. '), toc

	%% Update U
	U_X(:,tidx) = U_X(:,tidx)+X(:,tidx)-Xhat(:,tidx);

	U_Fu_nuc(:,tidx) = U_Fu_nuc(:,tidx)+Fu_nuc(:,tidx)-Fu(:,tidx);
	U_Fk_nuc(:,tidx) = U_Fk_nuc(:,tidx)+Fk_nuc(:,tidx)-Fk(:,tidx);

	%U_Fu_TF = U_Fu_TF+Fu_TF-Fu;
	%U_Fk_TF = U_Fk_TF+Fk_TF-Fk;

	U_Fu_nn(:,tidx) = U_Fu_nn(:,tidx)+Fu_nn(:,tidx)-Fu(:,tidx);
	U_Fk_nn(:,tidx) = U_Fk_nn(:,tidx)+Fk_nn(:,tidx)-Fk(:,tidx);

	U_Su_nn = U_Su_nn+Su_nn-Su;
	U_Sk_nn = U_Sk_nn+Sk_nn-Sk;


	%% Compute loss
	loss(iter) = lossfun;
	loss_aug(iter) = lossfun_aug;
	if exist('groundtruth','var') && ~isempty(groundtruth),
		loss_gt(iter) = lossfun_gt;
	else
		loss_gt(iter) = 0;
	end
	fprintf('[Iter: %d] Loss: %f, Loss aug: %f, Loss gt: %f\n\n',iter,loss(iter),loss_aug(iter),loss_gt(iter))
	tvec = 2:iter;
	subplot(221)
	plot(tvec,loss(tvec:iter))
	title('loss')
	subplot(222)
	plot(tvec,loss_aug(tvec))
	title('augmented loss')
	subplot(223)
	imagesc(Sk'*Sk0)
	title('gt Sk correlation')
	subplot(224)
	imagesc(Fk*Fk0')
	title('gt Fk correlation')
	drawnow

end
keyboard


	function Pmotioncorrect = Pu(it)
		Pmotioncorrect = P0;
	end
	function Pmotioncorrect = Pk(it)
		Pmotioncorrect = P0(:,mask);
	end

%%
	function l = lossfun;
		l = 0;

		% primary loss: the KL divergence between Y and Xhat
		l_Poiss = Y(:,tidx2).*(log(Y(:,tidx2))-log(Xhat_nn(:,tidx2)));
		l_Poiss = sum(l_Poiss(isfinite(l_Poiss))) + sum(sum(Xhat_nn(:,tidx2)-Y(:,tidx2)));

		% if (l_Poiss < 0) || ~isfinite(l_Poiss),
		% 	keyboard
		% end

		% regularizers
		l_Fk_nuc = lambdaFk_nuc*sum(svd(Fk));
		l_Fu_nuc = lambdaFu_nuc*sum(svd(Fu));

		l = l_Poiss + l_Fk_nuc + l_Fu_nuc;

		disp(['l_Poiss: ' num2str(l_Poiss)])
		disp(['l_Fk_nuc: ' num2str(l_Fk_nuc)])
		disp(['l_Fu_nuc: ' num2str(l_Fu_nuc)])

		l = l/numel(Y(:,tidx));
	end
	function l_aug = lossfun_aug;
		l_aug = 0;

		% primary loss: the KL divergence between Y and X
		l_Poiss = Y(:,tidx2).*(log(Y(:,tidx2))-log(X(:,tidx2)));
		l_Poiss = sum(l_Poiss(isfinite(l_Poiss))) + sum(sum(X(:,tidx2)-Y(:,tidx2)));
		l_X = (rho/2)*norm(Xhat(:,tidx2)-X(:,tidx2)+U_X(:,tidx2),'fro');

		l_Fk = 0;
		l_Fk = l_Fk + lambdaFk_nuc*sum(svd(Fk_nuc));
		l_Fk = l_Fk + (rho/2)*norm(Fk_nuc(:)-Fk(:)+U_Fk_nuc(:));
		l_Fk = l_Fk + (rho/2)*norm(Fk_nn(:)-Fk(:)+U_Fk_nn(:));
		% l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)
		% l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)
		l_Fu = 0;
		l_Fu = l_Fu + lambdaFu_nuc*sum(svd(Fu_nuc));
		l_Fu = l_Fu + (rho/2)*norm(Fu_nuc(:)-Fu(:)+U_Fu_nuc(:));
		l_Fu = l_Fu + (rho/2)*norm(Fu_nn(:)-Fu(:)+U_Fu_nn(:));
		l_Sk = 0;
		l_Sk = l_Sk + (rho/2)*norm(Sk_nn(:)-Sk(:)+U_Sk_nn(:));
		l_Su = 0;
		l_Su = l_Su + (rho/2)*norm(Su_nn(:)-Su(:)+U_Su_nn(:));

		l_aug = l_Poiss + l_X + l_Fk + l_Fu + l_Sk + l_Su;

		disp(['l_Poiss: ' num2str(l_Poiss)])
		disp(['l_X: ' num2str(l_X)])
		disp(['l_Fk: ' num2str(l_Fk)])
		disp(['l_Fu: ' num2str(l_Fu)])
		disp(['l_Sk: ' num2str(l_Sk)])
		disp(['l_Su: ' num2str(l_Su)])

		l_aug = l_aug/numel(Y(:,tidx));
	end
	function l_gt = lossfun_gt;
		l_gt = norm(Fk(:,tidx)-Fk0(:,tidx),'fro') + norm(Sk(:)-Sk0(:));
	end



end