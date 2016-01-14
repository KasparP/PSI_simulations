# from numba import jit
import numpy as np
import scipy as sp
from scipy import io
from numpy import ndarray
import time
import random
import matplotlib.pyplot as plt
import progressbar
import os
# import theano
# import theano.tensor as TT

# @jit


def Pu(it):
    return P0  # function to motioncorrect P at a given frame


def Pk(it):
    return P0[:, mask]

# IS THIS THE CORRECT SIZE TO INITIALIZE THESE?


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


def reconstruct_cpu(Y,Sk,Fk,Su,Fu,Nframes,nIter,eta,mu,adagrad,groundtruth=None, out_fn = 'output.mat'):
    print 'Working in: ', os.getcwd()
    print 'Y (min,max): (%f,%f)' % (Y.min(),Y.max())
    Nproj, T = Y.shape
    Nvox, Nsk = Sk.shape
    Nvox, Nsu = Su.shape

    loss_gt = np.zeros((nIter))
    loss = np.zeros((nIter))
    dSu = np.empty(Su.shape)
    dSk = np.empty(Sk.shape)
    dFu = np.empty(Fu.shape)
    dFk = np.empty(Fk.shape)
    deltaSu = np.zeros(Su.shape)
    deltaSk = np.zeros(Sk.shape)
    deltaFu = np.zeros(Fu.shape)
    deltaFk = np.zeros(Fk.shape)
    etaSu = np.empty(Su.shape)
    etaSk = np.empty(Sk.shape)
    etaFu = np.empty(Fu.shape)
    etaFk = np.empty(Fk.shape)
    etaSu2 = np.zeros(Su.shape)
    etaSk2 = np.zeros(Sk.shape)
    etaFu2 = np.zeros(Fu.shape)
    etaFk2 = np.zeros(Fk.shape)
    Xhat = np.zeros(Y.shape)
    PSFu = np.zeros(Y.shape)
    PSFk = np.zeros(Y.shape)
    E = np.zeros(Y.shape)

    for ITER in range(nIter):

        # subsample in time
        if Nframes < T:
            tidx2 = random.sample(range(0, T), Nframes)
        else:
            tidx2 = range(T)
        if (ITER % 100) == 0:
            tidx = range(T)
        else:
            tidx = tidx2

        print ('Pre-computing P*[Sk*Fk+Su*Fu]... ')
        tic = time.time()
        # b=0;
        #bar = progressbar.ProgressBar()
        for it in tidx: #bar(tidx):
            # if it == tidx[0]:
                # print type(Fk)
                # print type(Fu)
                # print type(Sk)
                # print type(Su)
                # print '-----'
                # print Pk(it).shape
                # print Sk.shape
                # print Pk(it).shape
                # print Sk.dot(Fk[:, it]).shape
                # print Pk(it).dot(np.reshape(Sk.dot(Fk[:, it]), (-1, 1))).shape
                # print np.ravel(Pk(it).dot(np.reshape(Sk.dot(Fk[:, it]), (-1, 1)))).shape
                # print PSFk[:,it].shape
                # print it
            PSFu[:, [it]] = Pu(it).dot(Su.dot(Fu[:, [it]]))
            #PSFk[:, it] = Pk(it).dot(Sk.dot(Fk[:, it]))
            PSFk[:, [it]] = np.ravel(Pk(it).dot(np.reshape(Sk.dot(Fk[:, [it]]), (-1, 1))))  #necessary because Sk is sparse matrix, not array. Could avoid ravel by converting all arrays to matrices?

        Xhat[:, tidx] = PSFu[:, tidx] + PSFk[:, tidx]
        E[:, tidx] = Y[:, tidx] - Xhat[:, tidx]
        print 'Done precompute. ', time.time() - tic, 'seconds'

        ## Compute loss
        # loss[ITER] = lossfun()
        loss[ITER] = 0.5*np.mean(np.square(E[:,tidx]).ravel())
        if groundtruth is not None:
            maxind = sp.minimum(10, len(tidx))
            recon = Su.dot(Fu[:,[tidx[0:maxind-1]]])  #kinda slow
            recon[mask,:] += Sk.dot(Fk[:,[tidx[0:maxind-1]]])
            print 'Reconstructed image norm: ', sp.linalg.norm(recon)
            recon_gt = ndarray.reshape(groundtruth.IM, (-1,1)) + groundtruth.Su.dot(groundtruth.Fu[:,tidx[0:maxind-1]]) + groundtruth.seg.dot(groundtruth.activity[:,tidx[0:maxind-1]]) #really slow
            print 'Ground truth image norm: ', sp.linalg.norm(recon_gt)
            loss_gt[ITER] = sp.linalg.norm(recon - recon_gt)/maxind

        print '[Iter: %d] Loss: %f' % (ITER,loss[ITER])
        if (ITER % 100) == 0:
            if groundtruth is None:
                io.savemat(out_fn,{'loss':loss,'Y':Y,'Xhat':Xhat,'Sk':Sk,'Fk':Fk,'Su':Su,'Fu':Fu})
            else:
                io.savemat(out_fn,{'loss':loss,'loss_gt':loss_gt, 'Y':Y,'Xhat':Xhat,'Sk':Sk,'Fk':Fk,'Su':Su,'Fu':Fu, 'recon':recon[:,0], 'recon_gt':recon_gt[:,0], 'IM':groundtruth.IM})


        # compute gradients
        print ('Computing gradients... ')
        tic = time.time()
        dSu.fill(0.0)
        dSk.fill(0.0)
        #bar = progressbar.ProgressBar()
        for it in tidx2: #bar(tidx2):
            dSu = dSu + np.outer((Pu(it).T).dot(E[:, it]),Fu[:,it])
            dSk = dSk + np.outer((Pk(it).T).dot(E[:, it]),Fk[:,it])
            dFu[:, it] = ((Pu(it).dot(Su)).T).dot(E[:, it])
            dFk[:, it] = ((Pk(it).dot(Sk)).T).dot(E[:, it])
        dSu = dSu/len(tidx2)
        dSk = dSk/len(tidx2)
        print 'Done. ', time.time() - tic, 'seconds'

        # update learning rate (Adagrad)
        if adagrad == True:
            etaSu2 = etaSu2 + np.square(dSu)
            etaSk2 = etaSk2 + np.square(dSk)
            etaFu2 = etaFu2 + np.square(np.mean(np.fabs(dFu[:,tidx2])))
            etaFk2 = etaFk2 + np.square(np.mean(np.fabs(dFk[:,tidx2])))
            # etaFu2[:,tidx2] = etaFu2[:,tidx2] + np.square(dFu[:,tidx2])
            # etaFk2[:,tidx2] = etaFk2[:,tidx2] + np.square(dFk[:,tidx2])

            etaSu = 1./(5e4+np.sqrt(etaSu2))
            etaSk = 1./(5e4+np.sqrt(etaSk2))
            etaFu = 1./(5e4+np.sqrt(etaFu2))
            etaFk = 1./(5e4+np.sqrt(etaFk2))
            # etaFu[:,tidx2] = 1./(1e4+np.sqrt(etaFu2[:,tidx2]))
            # etaFk[:,tidx2] = 1./(1e4+np.sqrt(etaFk2[:,tidx2]))

            # compute updates, with momentum
            deltaSu = mu*deltaSu + eta*etaSu*dSu
            deltaSk = mu*deltaSk + eta*etaSk*dSk
            deltaFu[:,tidx2] = mu*deltaFu[:,tidx2] + eta*etaFu[:,tidx2]*dFu[:,tidx2]
            deltaFk[:,tidx2] = mu*deltaFk[:,tidx2] + eta*etaFk[:,tidx2]*dFk[:,tidx2]

        else:
            etaSu = 1
            etaSk = 1
            etaFu = 1
            etaFk = 1

            # compute updates
            deltaSu = eta*etaSu*dSu
            deltaSk = eta*etaSk*dSk
            deltaFu[:,tidx2] = eta*etaFu*dFu[:,tidx2]
            deltaFk[:,tidx2] = eta*etaFk*dFk[:,tidx2]

        print 'mean (stepsize,eta) Su = (%f , %f)' % (np.mean(np.fabs(deltaSu.ravel())),np.mean(etaSu))
        print 'mean (stepsize,eta) Sk = (%f , %f)' % (np.mean(np.fabs(deltaSk.ravel())),np.mean(etaSk))
        print 'mean (stepsize,eta) Fu[:,it] = (%f , %f)' % (np.mean(np.fabs(deltaFu[:,it].ravel())),np.mean(etaFu))
        print 'mean (stepsize,eta) Fk[:,it] = (%f , %f)' % (np.mean(np.fabs(deltaFk[:,it].ravel())),np.mean(etaFk))

        # clip gradients if they are too big
        np.clip(deltaSu,-1e3,1e3,out=deltaSu)
        np.clip(deltaSk,-1e3,1e3,out=deltaSk)
        np.clip(deltaFu,-1e3,1e3,out=deltaFu)
        np.clip(deltaFk,-1e3,1e3,out=deltaFk)

        # update
        Su = Su + deltaSu
        Sk = Sk + deltaSk
        Fu[:,tidx2] = Fu[:,tidx2] + deltaFu[:,tidx2]
        Fk[:,tidx2] = Fk[:,tidx2] + deltaFk[:,tidx2]

        # rectify, and normalize to 1
        Fu[:,tidx2] = np.maximum(0, Fu[:,tidx2])
        Fk[:,tidx2] = np.maximum(0, Fk[:,tidx2])
        Su = np.maximum(0, Su)
        Sk = np.maximum(0, Sk)
        Su = Su * 1 / np.sum(np.finfo(np.float).eps+Su, 0)
        Sk = Sk * 1 / np.sum(np.finfo(np.float).eps+Sk, 0)

        print "\n"

    return (loss, Sk, Fk, Su, Fu)


def reconstruct_theano(Y,Sk,Fk,Su,Fu):

    # Set up a matrix factorization problem to optimize.
    TY = theano.shared(Y.astype(theano.config.floatX), name='Y')
    TSk = theano.shared(Sk.astype(theano.config.floatX), name='Sk')
    TFk = theano.shared(Fk.astype(theano.config.floatX), name='Fk')
    TSu = theano.shared(Su.astype(theano.config.floatX), name='Su')
    TFu = theano.shared(Fu.astype(theano.config.floatX), name='Fu')
    Tloss = 0.5*TT.sqr(TY - (TT.dot(TSk, TFk) + TT.dot(TSu, TFu))).sum()

    # Minimize the regularized loss with respect to a data matrix.
    y = np.dot(rand(A, K), rand(K, B)) + rand(A, B)

    return (loss, Sk, Fk, Su, Fu)


def prepexpt(fn = '../Problem_nonoise_v1.mat'):
    global P0
    global mask
    # fn = '../PSI/Problem_nonoise_v1.mat'
    # fn = '../../PSI/PRECOMP_nonoise_py.mat'

    # def initialize(fn):
    D = io.loadmat(fn, struct_as_record=False, squeeze_me=True)
    # print type(D['ground_truth']['seg'])
    # reconstruct_imaging_ADMM(D['obs'],D['opts'],1000,40,D['ground_truth'])
    obs = D['obs']
    opts = D['opts']
    groundtruth = D['ground_truth']

    # NOTES:
    # indexing is currently 1-based for many things, this needs to be fixed!

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

    [Nvox, Nproj] = opts.P.shape
    [Nproj, T] = obs.data_in.shape

    Sk0 = groundtruth.seg[mask, :]
    Fk0 = groundtruth.activity[:, :T]

    print('Initializing variables...')
    tic = time.time()

    Y = obs.data_in
    P0 = opts.P.T  # transpose

    #Nsk = groundtruth.seg.shape[1]
    #Sk = 1e-3 * sp.rand(Nvoxk, Nsk)  # +Sk0
    Sk = D['S_init']
    Sk = Sk[mask,:]

    Nsk = Sk.shape[1]
    Nsu = groundtruth.unsuspectedPos.shape[1] #this can be varied

    print '#Sk:', Nsk, '#Su:', Nsu

    Fk = 5e-1 * sp.rand(Nsk, T)  # +Fk0

    Su = 1e-3 * sp.rand(Nvox, Nsu)
    Fu = 5e-2 * sp.rand(Nsu, T)

    print 'Done initialization!', time.time() - tic, 'seconds'

    return (Y,Sk,Fk,Su,Fu,groundtruth)
