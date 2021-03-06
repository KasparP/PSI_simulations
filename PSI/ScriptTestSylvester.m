Nproj = 100;
Nvox = 25*25;
Nsrc = 10;
Nt = 100;

% %% Test solve_A_X_BXC.m

% A = rand(Nvox,Nsrc);
% X = randn(Nvox,Nsrc);
% B = randn(Nvox,Nvox);
% C = randn(Nsrc,Nsrc);

% Xhat = solve_A_X_BXC(A,X+B*X*C);
% norm(X-Xhat,'fro')

fprintf('Generating fake parameters...'),tic
P = rand(Nproj,Nvox,Nt);
Fu = rand(Nsrc,Nt);
Su = rand(Nvox,Nsrc);
Fk = rand(Nsrc,Nt);
Sk = rand(Nvox,Nsrc);
toc

fprintf('Generating fake projection data...'),tic
X = zeros(Nproj,Nt);
for it = 1:Nt,
	X(:,it) = P(:,:,it)*Su*Fu(:,it);
end
toc

fprintf('Generating fake Sylvester data...'),tic
A = rand(Nvox,Nvox,Nt);
B = rand(Nsrc,Nsrc,Nt);
XS = prox_matrix(rand(Nvox,Nsrc),2,@prox_maxk);
C = zeros(Nvox,Nsrc);
for it = 1:Nt,
	% A(:,:,it) = P(:,:,it)'*P(:,:,it);
	% B(:,:,it) = Fu(:,it)*Fu(:,it)';
	C = C + A(:,:,it)*XS*B(:,:,it)';
end
toc

fprintf('Solving the Sylvester system...'),tic
Xhat = gensylvester(A,B,C,XS+rand(Nvox,Nsrc));
subplot(211),imagesc(XS')
subplot(212),imagesc(Xhat')
toc

