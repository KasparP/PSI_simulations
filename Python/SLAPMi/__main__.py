from numba import jit
import numpy as np
import scipy as sp
from scipy import io
from numpy import ndarray
import time
import random
import matplotlib.pyplot as plt
import progressbar

# @jit

# fn = '../PSI/Problem_nonoise_v1.mat'
fn = '../../PSI/PRECOMP_nonoise_py.mat'

# def initialize(fn):
D = io.loadmat(fn, struct_as_record=False, squeeze_me=True)
# print type(D['ground_truth']['seg'])
# reconstruct_imaging_ADMM(D['obs'],D['opts'],1000,40,D['ground_truth'])
obs = D['obs']
opts = D['opts']
nIter = 1000
Nframes = 2
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

if Nframes < T:
    tidx = random.sample(range(0, T), Nframes)
else:
    tidx = range(0, T)  # 1:T
tidx2 = range(0, T)

print 'Done initialization!', time.time() - tic, 'seconds'

#Function definitions
def lossfun():
    l = 0

    # primary loss: the KL divergence between Y and Xhat
    l_Poiss = Y[:, tidx2] * (np.log(Y[:, tidx2]) - np.log(Xhat_nn[:, tidx2]))
    l_Poiss = sum(l_Poiss[np.isfinite(l_Poiss)]) + sum(sum(Xhat_nn[:, tidx2] - Y[:, tidx2]))

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
    l_Poiss = sum(l_Poiss[np.isfinite(l_Poiss)]) + sum(sum(X[:, tidx2] - Y[:, tidx2]))
    l_X = (rho / 2) * sum(sum((Xhat[:, tidx2] - X[:, tidx2] + np.square(U_X[:, tidx2]) - np.square(U_X[:, tidx2]))))

    l_Fk = 0
    [u, s, v] = np.linalg.svd(Fk_nuc)
    l_Fk = l_Fk + lambdaFk_nuc * sum(s)
    l_Fk = l_Fk + (rho / 2) * sum(
        (ndarray.flatten(Fk_nuc) - ndarray.flatten(Fk) + np.square(ndarray.flatten(U_Fk_nuc)) - np.square(ndarray.flatten(U_Fk_nuc))))
    l_Fk = l_Fk + (rho / 2) * sum(
        (ndarray.flatten(Fk_nn) - ndarray.flatten(Fk) + np.square(ndarray.flatten(U_Fk_nn)) - np.square(ndarray.flatten(U_Fk_nn))))
    # l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)
    # l_Fk = l_Fk + lambdaFk_TF*sum(Fk_TF*)

    l_Fu = 0
    [u, s, v] = np.linalg.svd(Fu_nuc)
    l_Fu = l_Fu + lambdaFu_nuc * sum(s)
    l_Fu = l_Fu + (rho / 2) * sum(
        (ndarray.flatten(Fu_nuc) - ndarray.flatten(Fu) + np.square(ndarray.flatten(U_Fu_nuc)) - np.square(ndarray.flatten(U_Fu_nuc))))
    l_Fu = l_Fu + (rho / 2) * sum(
        (ndarray.flatten(Fu_nn) - ndarray.flatten(Fu) + np.square(ndarray.flatten(U_Fu_nn)) - np.square(ndarray.flatten(U_Fu_nn))))

    l_Sk = 0
    l_Sk = l_Sk + (rho / 2) * sum(
        (ndarray.flatten(Sk_nn) - ndarray.flatten(Sk) + np.square(ndarray.flatten(U_Sk_nn)) - np.square(ndarray.flatten(U_Sk_nn))))
    l_Su = 0
    l_Su = l_Su + (rho / 2) * sum(
        (ndarray.flatten(Su_nn) - ndarray.flatten(Su) + np.square(ndarray.flatten(U_Su_nn)) - np.square(ndarray.flatten(U_Su_nn))))

    l_aug = l_Poiss + l_X + l_Fk + l_Fu + l_Sk + l_Su

    print 'l_Poiss: ', l_Poiss
    print 'l_X: ', l_X
    print 'l_Fk: ', l_Fk
    print 'l_Fu: ', l_Fu
    print 'l_Sk: ', l_Sk
    print 'l_Su: ', l_Su

    l_aug = l_aug / Y[:, tidx2].size
    return l_aug

def lossfun_gt():  # WE NEED A PROPER GT LOSS FUNCTION! Previous one was sensitive to swapping of Sk's
    l_gt = 10
    return l_gt

eta = 0.1
loss = np.zeros((nIter))
dSu = np.empty(Su.shape)
dSk = np.empty(Sk.shape)
dFu = np.empty(Fu.shape)
dFk = np.empty(Fk.shape)
etaSu = np.empty(Su.shape)
etaSk = np.empty(Sk.shape)
etaFu = np.empty(Fu.shape)
etaFk = np.empty(Fk.shape)
etaSu2 = np.empty(Su.shape)
etaSk2 = np.empty(Sk.shape)
etaFu2 = np.empty(Fu.shape)
etaFk2 = np.empty(Fk.shape)
Xhat = np.zeros(Y.shape)
PSFu = np.zeros(Y.shape)
PSFk = np.zeros(Y.shape)
E = np.zeros(Y.shape)
for ITER in range(nIter):
    # subsample in time
    if Nframes < T:
        tidx = random.sample(range(0, T), Nframes)
        tidx2 = tidx
        if (ITER % 50) == 0:
            tidx2 = range(0, T)


    print ('Pre-computing P*[Sk*Fk+Su*Fu]... ')
    tic = time.time()
    # b=0;
    bar = progressbar.ProgressBar()
    for it in bar(tidx2):
        # print it, '\r'
        PSFu[:, it] = Pu(it).dot(Su.dot(Fu[:, it]))
        PSFk[:, it] = Pk(it).dot(Sk.dot(Fk[:, it]))
    # fprintf([repmat('\b',1,b)]); b=fprintf('%d',it);

    Xhat[:, tidx2] = PSFu[:, tidx2] + PSFk[:, tidx2]
    E[:, tidx2] = Y[:, tidx2] - Xhat[:, tidx2]
    print 'Done precompute. ', time.time() - tic, 'seconds'

    # compute gradients
    print ('Computing gradients... ')
    tic = time.time()
    dSu.fill(0.0)
    dSk.fill(0.0)
    bar = progressbar.ProgressBar()
    for it in bar(tidx2):
        dFu[:, it] = ((Pu(it).dot(Su)).T).dot(E[:, it])
        dSu = dSu + np.outer((Pu(it).T).dot(E[:, it]),Fu[:,it])
        dFk[:, it] = ((Pk(it).dot(Sk)).T).dot(E[:, it])
        dSk = dSk + np.outer((Pk(it).T).dot(E[:, it]),Fk[:,it])
    dSu = dSu/len(tidx2)
    dSk = dSk/len(tidx2)
    print 'Done. ', time.time() - tic, 'seconds'

    # update learning rate
    etaSu2 = etaSu2 + np.square(dSu)
    etaSk2 = etaSk2 + np.square(dSk)
    etaFu2[:,tidx2] = etaFu2[:,tidx2] + np.square(dFu[:,tidx2])
    etaFk2[:,tidx2] = etaFk2[:,tidx2] + np.square(dFk[:,tidx2])

    etaSu = 1/(1+np.sqrt(etaSu2))
    etaSk = 1/(1+np.sqrt(etaSk2))
    etaFu[:,tidx2] = 1/(1+np.sqrt(etaFu2[:,tidx2]))
    etaFk[:,tidx2] = 1/(1+np.sqrt(etaFk2[:,tidx2]))

    # update
    Su = Su - eta*etaSu*dSu
    Sk = Sk - eta*etaSk*dSk
    Fu[:,tidx2] = Fu[:,tidx2] - eta*etaFu[:,tidx2]*dFu[:,tidx2]
    Fk[:,tidx2] = Fk[:,tidx2] - eta*etaFk[:,tidx2]*dFk[:,tidx2]

    # rectify, and normalize to 1
    Fu[:,tidx2] = np.maximum(0, Fu[:,tidx2])
    Fk[:,tidx2] = np.maximum(0, Fk[:,tidx2])
    Su = np.maximum(0, Su)
    Sk = np.maximum(0, Sk)
    Su = Su * 1 / np.sum(Su, 0)
    Sk = Sk * 1 / np.sum(Sk, 0)



    ## Compute loss
    # loss[ITER] = lossfun()
    loss[ITER] = np.linalg.norm(E.ravel())


    print '[Iter: %d] Loss: %f,\n' % (ITER,loss[ITER])

    # tvec = range(2, ITER + 1)
    # tvec2 = np.arange(51, ITER, 50).tolist()
    # plt.subplot(211)
    # plt.plot(tvec, loss[tvec], tvec2, loss[tvec2])
    # plt.title('loss')
    # plt.subplot(212)
    # plt.plot(tvec, loss_aug[tvec], tvec2, loss_aug[tvec2])
    # plt.title('augmented loss')
    # plt.subplot(223)
    # imagesc(Sk'*Sk0)
    # title('gt Sk correlation')
    # subplot(224)
    # imagesc(Fk*Fk0')
    # title('gt Fk correlation')

    # plt.pause(0.0001)  # ensure this gets drawn

if __name__ == '__main__':
    initialize(binLocation)
