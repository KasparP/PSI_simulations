import numpy as np
import scipy as sp
from scipy import io
from numpy import ndarray
import time
import random
import matplotlib.pyplot as plt

fn = 'C:\Users\podgorskik\Documents\GitHub\PSI_simulations\PSI\PRECOMP_nonoise_py'


def lossfun():
    l = 0

    # primary loss: the KL divergence between Y and Xhat
    l_Poiss = Y[:, tidx2] * (np.log(Y[:, tidx2]) - np.log(Xhat_nn[:, tidx2]))
    l_Poiss = sum(l_Poiss[np.isfinite(l_Poiss)]) + sum(Xhat_nn[:, tidx2] - Y[:, tidx2])

    if (l_Poiss < 0) or ~np.isfinite(l_Poiss):
        print 'LOSSFUN produced an error'

    # regularizers
    # NOTE: Do we want to sum only a subset of s? -KP
    [u, s, v] = np.linalg.svd(Fk)
    l_Fk_nuc = lambdaFk_nuc * sum(s)
    [u, s, v] = np.linalg.svd(Fu)
    l_Fu_nuc = lambdaFu_nuc * sum(s)

    l = l_Poiss + l_Fk_nuc + l_Fu_nuc

    print 'l_Poiss: ', l_Poiss
    print 'l_Fk_nuc: ', l_Fk_nuc
    print 'l_Fu_nuc: ', l_Fu_nuc

    l = l / Y[:, tidx2].size

    return 1


def lossfun_aug():
    l_aug = 0

    # primary loss: the KL divergence between Y and X
    l_Poiss = Y[:, tidx2] * (np.log(Y[:, tidx2]) - np.log(X[:, tidx2]))
    l_Poiss = sum(l_Poiss[np.isfinite(l_Poiss)]) + sum(X[:, tidx2] - Y[:, tidx2])
    l_X = (rho / 2) * sum((Xhat[:, tidx2] - X[:, tidx2] + np.square(U_X[:, tidx2]) - np.square(U_X[:, tidx2])))

    l_Fk = 0
    [u, s, v] = np.linalg.svd(Fk_nuc)
    l_Fk = l_Fk + lambdaFk_nuc * sum(s)
    l_Fk = l_Fk + (rho / 2) * sum(
        (np.flatten(Fk_nuc) - np.flatten(Fk) + np.square(np.flatten(U_Fk_nuc)) - np.square(np.flatten(U_Fk_nuc))))
    l_Fk = l_Fk + (rho / 2) * sum(
        (np.flatten(Fk_nn) - np.flatten(Fk) + np.square(np.flatten(U_Fk_nn)) - np.square(np.flatten(U_Fk_nn))))
    # l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)
    # l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)

    l_Fu = 0
    [u, s, v] = np.linalg.svd(Fu_nuc)
    l_Fu = l_Fu + lambdaFu_nuc * sum(s)
    l_Fu = l_Fu + (rho / 2) * sum(
        (np.flatten(Fu_nuc) - np.flatten(Fu) + np.square(np.flatten(U_Fu_nuc)) - np.square(np.flatten(U_Fu_nuc))))
    l_Fu = l_Fu + (rho / 2) * sum(
        (np.flatten(Fu_nn) - np.flatten(Fu) + np.square(np.flatten(U_Fu_nn)) - np.square(np.flatten(U_Fu_nn))))

    l_Sk = 0
    l_Sk = l_Sk + (rho / 2) * sum(
        (np.flatten(Sk_nn) - np.flatten(Sk) + np.square(np.flatten(U_Sk_nn)) - np.square(np.flatten(U_Sk_nn))))
    l_Su = 0
    l_Su = l_Su + (rho / 2) * sum(
        (np.flatten(Su_nn) - np.flatten(Su) + np.square(np.flatten(U_Su_nn)) - np.square(np.flatten(U_Su_nn))))

    l_aug = l_Poiss + l_X + l_Fk + l_Fu + l_Sk + l_Su

    print 'l_Poiss: ', l_Poiss
    print 'l_X: ', l_X
    print 'l_Fk: ', l_Fk
    print 'l_Fu: ', l_Fu
    print 'l_Sk: ', l_Sk
    print 'l_Su: ', l_Su

    l_aug = l_aug / Y[:, tidx2].size
    return l_aug


def lossfun_gt():  # WE NEED A PROPER GT LOSS FUNCTION!
    l_gt = np.linalg.norm(Fk[:, tidx] - Fk0[:, tidx], 'fro') + np.linalg.norm(np.flatten(Sk) - np.flatten(Sk0))
    return l_gt


# def initialize(fn):
D = io.loadmat(fn, struct_as_record=False, squeeze_me=True)
# print type(D['ground_truth']['seg'])
# reconstruct_imaging_ADMM(D['obs'],D['opts'],1000,40,D['ground_truth'])
obs = D['obs']
opts = D['opts']
nIter = 1000
Nframes = 40
groundtruth = D['ground_truth']

# NOTES:
# indexing is currently 1-based for many things, this needs to be fixed!

# def reconstruct_imaging_ADMM(obs, opts, nIter, Nframes, groundtruth):

rho = 1e-0
lambdaFk_nuc = 1e-10
lambdaFu_nuc = 1e0
lambdaFk_TF = 1e-10
lambdaFu_TF = 1e-10

bw = groundtruth.bw

mask = ndarray.flatten(bw) > 0
Nvoxk = sum(mask)

print(opts.P.shape)
obs.data_in = opts.P[mask, :].T.dot(groundtruth.seg[mask, :]).dot(groundtruth.activity)

[Nvox, Np] = opts.P.shape
[Np, T] = obs.data_in.shape
Nsk = groundtruth.activity.shape[0]
Nsu = groundtruth.unsuspectedPos.shape[1]

Sk0 = groundtruth.seg[mask, :]
Fk0 = groundtruth.activity[:, :T]

print('Initializing variables...')
tic = time.time()

Y = obs.data_in
P0 = opts.P.T  # transpose

X = sp.zeros((Np, T))
Xhat = sp.zeros((Np, T))
Xhat_nn = sp.zeros((Np, T))

# TODO: improve initialization!
Sk = 1e-3 * sp.rand(Nvoxk, Nsk)  # +Sk0
Fk = 5e-1 * sp.rand(Nsk, T)  # +Fk0

Su = 1e-3 * sp.rand(Nvox, Nsu)
Fu = 5e-2 * sp.rand(Nsu, T)

Fu_nuc = Fu  # zeros(size(Fu));
Fu_nn = Fu  # zeros(size(Fu));
Fu_TF = Fu  # zeros(size(Fu));
Fu_TF_z = Fu  # zeros(size(Fu));
Fu_TF_u = Fu  # zeros(size(Fu));
Fk_nuc = Fk  # zeros(size(Fk));
Fk_nn = Fk  # zeros(size(Fk));
Fk_TF = Fk  # zeros(size(Fk));
Fk_TF_z = Fk  # zeros(size(Fk));
Fk_TF_u = Fk  # zeros(size(Fk));

U_X = 0 * sp.ones((Np, T))
U_Fu_nuc = 0 * sp.ones(Fu.shape)
U_Fu_TF = 0 * sp.ones(Fu.shape)
U_Fu_nn = 0 * sp.ones(Fu.shape)
U_Fk_nuc = 0 * sp.ones(Fk.shape)
U_Fk_TF = 0 * sp.ones(Fk.shape)
U_Fk_nn = 0 * sp.ones(Fk.shape)
U_Su_nn = 0 * sp.ones(Su.shape)
U_Sk_nn = 0 * sp.ones(Sk.shape)


def Pu(it):
    return P0  # function to motioncorrect P at a given frame


def Pk(it):
    return P0[:, mask]


# IS THIS THE CORRECT SIZE TO INITIALIZE THESE?
PSFu = sp.zeros(Y.shape)
PSFk = sp.zeros(Y.shape)
U_X_PSFk = sp.zeros(PSFk.shape)
U_X_PSFu = sp.zeros(PSFu.shape)
loss = sp.zeros((1, nIter))
loss_aug = sp.zeros((1, nIter))
loss_gt = sp.zeros((1, nIter))

print 'Done initialization!', time.time() - tic, 'seconds'

if Nframes < T:
    tidx = random.sample(range(1, T + 1), Nframes)
else:
    tidx = range(1, T + 1)  # 1:T

tidx2 = range(1, T + 1)
for ITER in range(1, nIter + 1):
    # subsample in time
    if Nframes < T:
        tidx = random.sample(range(1, T + 1), Nframes)
        tidx2 = tidx
        if ITER == 1 or (ITER % 50) == 0:
            tidx2 = range(1, T + 1)
    print ('Pre-computing P*[Sk*Fk+Su*Fu]... ')
    tic = time.time()
    # b=0;



    for it in tidx2:
        PSFu[:, it] = Pu(it).dot(Su.dot(Fu[:, it]))
        PSFk[:, it] = Pk(it).dot(Sk.dot(Fk[:, it]))
        Xhat_nn[:, it] = Pu(it).dot(np.maximum(Su, 0)).dot(np.maximum(Fu[:, it], 0)) + Pk(it).dot((np.maximum(Sk, 0)).dot(np.maximum(Fk[:, it], 0)))
    # fprintf([repmat('\b',1,b)]); b=fprintf('%d',it);

    Xhat[:, tidx2] = PSFu[:, tidx2] + PSFk[:, tidx2]  # no divide by 2?
    print 'Done precompute. ', time.time() - tic, 'seconds'

    X[:, tidx2] = prox_DKL(Y[:, tidx2], Xhat[:, tidx2] - U_X[:, tidx2], rho)
    U_X_PSFk[:, tidx2] = U_X[:, tidx2] + X[:, tidx2] - PSFk[:, tidx2]
    U_X_PSFu[:, tidx2] = U_X[:, tidx2] + X[:, tidx2] - PSFu[:, tidx2]

    #    Fu_nuc = prox_matrix(Fu-U_Fu_nuc,lambdaFu_nuc/rho,prox_l1)
    #    Fk_nuc = prox_matrix(Fk-U_Fk_nuc,lambdaFk_nuc/rho,prox_l1)


    # [Fu_TF, Fu_TF_z, Fu_TF_u] = prox_matrix_L1(D,Fu-U_Fu_TF,lambdaFu_TF,rho,1,10,Fu_TF,Fu_TF_z,Fu_TF_u);
    # [Fk_TF, Fk_TF_z, Fk_TF_u] = prox_matrix_L1(D,Fk-U_Fk_TF,lambdaFk_TF,rho,1,10,Fk_TF,Fk_TF_z,Fk_TF_u);

    Fu_nn[:, tidx2] = np.maximum(0, Fu[:, tidx2] - U_Fu_nn[:, tidx2])
    Fk_nn[:, tidx2] = np.maximum(0, Fk[:, tidx2] - U_Fk_nn[:, tidx2])

    # rectify, and normalize to 1
    Su_nn = np.maximum(0, Su - U_Su_nn)
    Su_nn = Su_nn * 1 / np.sum(Su_nn, 0)  # bsxfun(@times,Su_nn,1./sum(Su_nn,1))
    Sk_nn = np.maximum(0, Sk - U_Sk_nn)
    Sk_nn = Sk_nn * 1 / np.sum(Sk_nn, 0)  # bsxfun(@times,Sk_nn,1./sum(Sk_nn,1))

    # Update consensus variables
    print 'Estimating F, t=', tic
    b = 0
    for it in tidx2:
        PSu = Pu(it).dot(Su)
        PSk = Pk(it).dot(Sk)
        Fu[:, it] = sp.linalg.solve(PSu.T.dot(PSu) + 2 * sp.eye(Nsu), (
        PSu.T.dot(U_X_PSFk[:, it]) + (U_Fu_nuc[:, it] + U_Fu_nuc[:, it]) + (Fu_nuc[:, it] + Fu_nn[:, it])))
        Fk[:, it] = sp.linalg.solve(PSk.T.dot(PSk) + 2 * sp.eye(Nsk), (
        PSk.T.dot(U_X_PSFu[:, it]) + (U_Fk_nuc[:, it] + U_Fk_nuc[:, it]) + (Fk_nuc[:, it] + Fk_nn[:, it])))
        print '\b' * 8
        print "'%08d'" % it

    print '. Done. '
    print 'Estimating S... \n'

    Cu = U_Su_nn + Su_nn
    tidx = range(1, T + 1)  # randsample(T,10);
    for it in tidx:  # 1:length(Tidx),
        Cu = Cu + Pu(it).T.dot(U_X_PSFk[:, it].dot(Fu[:, it].T))
    Su = solve_S_Pfun(Pu, Fu, Cu, Su, tidx)

    Ck = U_Sk_nn + Sk_nn
    for it in tidx:  # 1:length(tidx),
        Ck = Ck + Pk(it).T.dot(U_X_PSFu[:, it].dot(Fk[:, it].T))
    Sk = solve_S_Pfun(Pk, Fk, Ck, Sk, tidx)

    print 'Done. '

    ## Update U
    U_X[:, tidx] = U_X[:, tidx] + X[:, tidx] - Xhat[:, tidx]

    U_Fu_nuc[:, tidx] = U_Fu_nuc[:, tidx] + Fu_nuc[:, tidx] - Fu[:, tidx]
    U_Fk_nuc[:, tidx] = U_Fk_nuc[:, tidx] + Fk_nuc[:, tidx] - Fk[:, tidx]

    # U_Fu_TF = U_Fu_TF+Fu_TF-Fu
    # U_Fk_TF = U_Fk_TF+Fk_TF-Fk

    U_Fu_nn[:, tidx] = U_Fu_nn[:, tidx] + Fu_nn[:, tidx] - Fu[:, tidx]
    U_Fk_nn[:, tidx] = U_Fk_nn[:, tidx] + Fk_nn[:, tidx] - Fk[:, tidx]

    U_Su_nn = U_Su_nn + Su_nn - Su
    U_Sk_nn = U_Sk_nn + Sk_nn - Sk

    ## Compute loss
    loss[ITER] = lossfun()
    loss_aug[ITER] = lossfun_aug()
    if 'groundtruth' in locals() and groundtruth.size > 0:
        loss_gt[ITER] = lossfun_gt()
    else:
        loss_gt[ITER] = 0

    # print '[Iter: %d] Loss: %f, Loss aug: %f, Loss gt: %f\n\n',iter,loss(iter),loss_aug(iter),loss_gt(iter))

    tvec = range(2, ITER + 1)
    tvec2 = np.arange(51, ITER, 50).tolist()
    plt.subplot(211)
    plt.plot(tvec, loss(tvec), tvec2, loss(tvec2))
    plt.title('loss')
    plt.subplot(212)
    plt.plot(tvec, loss_aug(tvec), tvec2, loss_aug(tvec2))
    plt.title('augmented loss')
    # plt.subplot(223)
    # imagesc(Sk'*Sk0)
    # title('gt Sk correlation')
    # subplot(224)
    # imagesc(Fk*Fk0')
    # title('gt Fk correlation')

    plt.pause(0.0001)  # ensure this gets drawn

if __name__ == '__main__':
    initialize(binLocation)
