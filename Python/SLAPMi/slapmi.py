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
import code
from NN_KP import NN_KP

# from pyspark import SparkConf, SparkContext
# import theano
# import theano.tensor as TT

# @jit

#os.environ['PYTHONPATH'] = (os.environ['PYTHONPATH'] + '/tier2/turaga/podgorskik/PSI_simulations/Python/SLAPMi/:')
from scipy import optimize

def Pu(P0,it):
    return P0  # function to motioncorrect P at a given frame

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


def reconstruct_cpu(Y,Sk,Fk,Su,Fu,P0,mask,masks,Nframes,nIter,eta,mu,adagrad,groundtruth=None, out_fn='output.mat', sparkcontext=None):
    print 'Working in: ', os.getcwd()
    print 'Y (min,max): (%f,%f)' % (Y.min(),Y.max())
    Nproj, T = Y.shape
    Nvox, Nsk = Sk.shape
    Nvox, Nsu = Su.shape

    loss_gt = np.zeros((nIter))
    loss = np.zeros((nIter))
    dSu = np.empty(Su.shape)

    dSk = sp.sparse.csc_matrix(Sk.shape) #np.empty(Sk.shape)

    dFu = np.empty(Fu.shape)
    dFk = np.empty(Fk.shape)
    deltaSu = np.empty(Su.shape)

    deltaSk = sp.sparse.csc_matrix(Sk.shape) #np.zeros(Sk.shape)

    deltaFu = np.zeros(Fu.shape)
    deltaFk = np.zeros(Fk.shape)
    etaSu = np.empty(Su.shape) #np.empty(Su.shape)

    etaSk = sp.sparse.csc_matrix(Sk.shape) #np.empty(Sk.shape)

    etaFu = np.empty(Fu.shape)
    etaFk = np.empty(Fk.shape)
    etaSu2 = np.empty(Su.shape)

    etaSk2 = sp.sparse.csc_matrix(Sk.shape)

    etaFu2 = np.zeros(Fu.shape)
    etaFk2 = np.zeros(Fk.shape)
    Xhat = np.zeros(Y.shape)
    PSFu = np.zeros(Y.shape)
    PSFk = np.zeros(Y.shape)
    E = np.zeros(Y.shape)


    Sk_nnz = Sk.nonzero()
    M = masks
    mask_ixs = [masks[:,x].nonzero()[0] for x in range(Sk.shape[1])]

    Su_in = Su.copy()
    Sk_in = Sk.copy()
    Fu_in = Fu.copy()
    Fk_in = Fk.copy()


    # code.interact(local=locals())


    #parallel solve of F
    if sparkcontext is not None:
        print 'Full solve of F...'
        Fk,Fu = solveFgivenS(Y, Sk, Su, masks, P0, sparkcontext)
        print 'F solve complete.'

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
            PSFu[:, it] = Pu(P0,it).dot(Su.dot(Fu[:, it]))
            PSFk[:, it] = Pu(P0,it).dot(Sk.dot(Fk[:, it]))
            #PSFk[:, [it]] = np.reshape(Pk(it).dot(np.reshape(Sk.dot(Fk[:, [it]]), (-1, 1))), (-1,1))  #necessary because Sk is sparse matrix, not array. Could avoid ravel by converting all arrays to matrices?

        Xhat[:, tidx] = PSFu[:, tidx] + PSFk[:, tidx]
        E[:, tidx] = Y[:, tidx] - Xhat[:, tidx]
        print 'Done precompute. ', time.time() - tic, 'seconds'

        ## Compute loss
        # loss[ITER] = lossfun()
        loss[ITER] = 0.5*np.mean(np.square(E[:,tidx]).ravel())
        print '[Iter: %d] Loss: %f' % (ITER,loss[ITER])
        print 'Ynorm: %f     Xnorm: %f' % (sp.linalg.norm(Y[:, tidx]), sp.linalg.norm(Xhat[:,tidx]))


        if groundtruth is not None:
            maxind = sp.minimum(10, len(tidx))
            recon = Su.dot(Fu[:,tidx[0:maxind]])  #kinda slow
            recon_U = recon.copy()
            recon_K = Sk.dot(Fk[:,tidx[0:maxind]])
            recon += recon_K

            recon_gt = ndarray.reshape(groundtruth.IM, (-1,1), order='F') + groundtruth.seg.dot(groundtruth.activity[:,tidx[0:maxind]]) #really slow
            gt_K = recon_gt.copy()
            gt_U = np.zeros(shape=(0, 1))
            if groundtruth.Su.size>1:
                gt_U = groundtruth.Su.dot(groundtruth.Fu[:,tidx[0:maxind]])
                recon_gt += gt_U

            for ix in range(maxind):
                P = Pu(P0,tidx[ix])
                discard, notP = np.logical_not(P.sum(0)).nonzero()
                recon[notP,ix] = 0
                recon_gt[notP,ix] = 0

            print 'Reconstructed image norm: ', sp.linalg.norm(recon)
            print 'Ground truth image norm: ', sp.linalg.norm(recon_gt)
            loss_gt[ITER] = sp.linalg.norm(recon - recon_gt)/np.sqrt(maxind)


        if (ITER % 100) == 0:
            print 'Saving data...'
            if groundtruth is None:
                io.savemat(out_fn,{'loss':loss,'Y':Y,'Xhat':Xhat,'Sk':Sk,'Fk':Fk,'Su':Su,'Fu':Fu, 'mask':mask})
            else:
                io.savemat(out_fn,{'loss':loss,'loss_gt':loss_gt, 'Y':Y,'Xhat':Xhat,'Sk':Sk,'Fk':Fk,'Su':Su,'Fu':Fu, 'recon':recon[:,0], 'recon_gt':recon_gt[:,0], 'recon_K':recon_K[:,0], 'recon_U':recon_U[:,0], 'gt_U':gt_U[:,0], 'gt_K':gt_K[:,0], 'IM':groundtruth.IM, 'mask':mask, 'PSFu':PSFu, 'PSFk':PSFk})


        # compute gradients
        print ('Computing gradients... ')
        tic = time.time()
        dSu = 0*dSu
        dSk = 0*dSk
        #bar = progressbar.ProgressBar()
        for it in tidx2: #bar(tidx2):
            Pt = Pu(P0, it)
            dSu = dSu + np.outer(Pt.T.dot(E[:, it]),Fu[:,it])

            PE = Pt.T.dot(E[:, it])  #projected error for this iter
            #code.interact(local = locals())

            dSk_tmp = sp.sparse.lil_matrix(Sk.shape)
            for S_ix in range(Sk.shape[1]):
                #A = (Pt.T.dot(E[:, it])[mask_ixs[S_ix]])*Fk[S_ix,it]
                #B = dSk_tmp
                #M = mask_ixs
                #code.interact(local = locals())
                dSk_tmp[mask_ixs[S_ix], S_ix] = ((PE[mask_ixs[S_ix]])*Fk[S_ix,it])[:,np.newaxis]
            dSk = dSk+dSk_tmp

            #dSk = dSk + np.outer((Pu(P0, it).T).dot(E[:, it]),Fk[:,it])
            dFu[:, it] = ((Pt.dot(Su)).T).dot(E[:, it])
            dFk[:, it] = ((Pt.dot(Sk)).T).dot(E[:, it])
        dSu = dSu/len(tidx2)
        dSk = dSk/len(tidx2)
        print 'Done. ', time.time() - tic, 'seconds'
        print 'dSu norm: ', sp.linalg.norm(dSu)

        #print 'dSk norm: ', sp.linalg.norm(dSk)

        # update learning rate (Adagrad)
        if adagrad == True:
            etaSu2 = etaSu2 + np.square(dSu)
            etaSk2 = etaSk2 + dSk.multiply(dSk)
            etaFu2 = etaFu2 + np.square(np.mean(np.fabs(dFu[:,tidx2])))
            etaFk2 = etaFk2 + np.square(np.mean(np.fabs(dFk[:,tidx2])))
            # etaFu2[:,tidx2] = etaFu2[:,tidx2] + np.square(dFu[:,tidx2])
            # etaFk2[:,tidx2] = etaFk2[:,tidx2] + np.square(dFk[:,tidx2])

            etaSu = 1./(5e4+np.sqrt(etaSu2))
            etaSk[Sk_nnz] = 1./(5e4+np.sqrt(etaSk2[Sk_nnz])) #need to keep this matrix sparse. This is slow.
            etaFu = 1./(5e4+np.sqrt(etaFu2))
            etaFk = 1./(5e4+np.sqrt(etaFk2))
            # etaFu[:,tidx2] = 1./(1e4+np.sqrt(etaFu2[:,tidx2]))
            # etaFk[:,tidx2] = 1./(1e4+np.sqrt(etaFk2[:,tidx2]))

            # compute updates, with momentum
            deltaSu = mu*deltaSu + eta*etaSu*dSu
            deltaSk = mu*deltaSk + eta*etaSk.multiply(dSk)
            deltaFu[:,tidx2] = mu*deltaFu[:,tidx2] + eta*etaFu[:,tidx2]*dFu[:,tidx2]
            deltaFk[:,tidx2] = mu*deltaFk[:,tidx2] + eta*etaFk[:,tidx2]*dFk[:,tidx2]

        else:
            etaSu = 1
            etaSk = 1
            etaFu = 1
            etaFk = 1

            # compute updates
            deltaSu = eta*etaSu*dSu
            deltaSk = eta*etaSk.multiply(dSk)
            deltaFu[:,tidx2] = eta*etaFu*dFu[:,tidx2]
            deltaFk[:,tidx2] = eta*etaFk*dFk[:,tidx2]

        print 'mean (stepsize,eta) Su = (%f , %f)' % (np.mean(np.fabs(deltaSu.ravel())),np.mean(etaSu))
        #print 'mean (stepsize,eta) Sk = (%f , %f)' % (np.mean(np.fabs(deltaSk.ravel())),np.mean(etaSk))
        print 'mean (stepsize,eta) Fu[:,it] = (%f , %f)' % (np.mean(np.fabs(deltaFu[:,it].ravel())),np.mean(etaFu))
        print 'mean (stepsize,eta) Fk[:,it] = (%f , %f)' % (np.mean(np.fabs(deltaFk[:,it].ravel())),np.mean(etaFk))

        # clip gradients if they are too big
        np.clip(deltaSu,-1e3,1e3,out=deltaSu)

        deltaSk[deltaSk>1e3] = 1e3
        deltaSk[deltaSk<-1e3] = -1e3

        #np.clip(deltaSk,-1e3,1e3,out=deltaSk)
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

        Sk[Sk<0] = 0

        scaleSu = np.sum(np.finfo(np.float).eps+Su, 0)
        scaleSk = np.asarray(Sk.sum(0)) + np.finfo(np.float).eps

        Su = Su * (1/scaleSu[np.newaxis,:])
        Sk[Sk_nnz] = Sk.multiply(1/scaleSk)[Sk_nnz]  #SLOW!!!!
        Fu = Fu * scaleSu[:,np.newaxis]
        Fk = np.multiply(scaleSk.T, Fk)

        #Full solve of F given the updated S


        print "\n"

    return (loss, Sk, Fk, Su, Fu, P0)


# def reconstruct_theano(Y,Sk,Fk,Su,Fu):
#
#     # Set up a matrix factorization problem to optimize.
#     TY = theano.shared(Y.astype(theano.config.floatX), name='Y')
#     TSk = theano.shared(Sk.astype(theano.config.floatX), name='Sk')
#     TFk = theano.shared(Fk.astype(theano.config.floatX), name='Fk')
#     TSu = theano.shared(Su.astype(theano.config.floatX), name='Su')
#     TFu = theano.shared(Fu.astype(theano.config.floatX), name='Fu')
#     Tloss = 0.5*TT.sqr(TY - (TT.dot(TSk, TFk) + TT.dot(TSu, TFu))).sum()
#
#     # Minimize the regularized loss with respect to a data matrix.
#     y = np.dot(rand(A, K), rand(K, B)) + rand(A, B)
#
#     return (loss, Sk, Fk, Su, Fu)


def prepexpt(fn = '../Problem_nonoise_v2_init.mat'):

    # fn = '../PSI/Problem_nonoise_v1.mat'
    # fn = '../../PSI/PRECOMP_nonoise_py.mat'

    D = io.loadmat(fn, struct_as_record=False, squeeze_me=True)

    rho = 1e-0
    lambdaFk_nuc = 1e-10
    lambdaFu_nuc = 1e0
    lambdaFk_TF = 1e-10
    lambdaFu_TF = 1e-10

    obs = D['obs']
    opts = D['opts']
    GT = D['GT']
    Sk = D['Sk']
    Fk = D['Fk']
    Su = D['Su']
    Fu = D['Fu']

    if len(Su.shape)<2:
        Su = Su[:,None]
        Fu = Fu[None,:]

    if len(GT.Su.shape)<2:
        GT.Su = GT.Su[:,None]
        GT.Fu = GT.Fu[None,:]

    masks = D['masks']

    bw = GT.bw
    mask = ndarray.flatten(bw) > 0
    Nvoxk = sum(mask)


    [Nvox, Nproj] = opts.P.shape
    [Nproj, T] = obs.data_in.shape

    #Sk0 = GT.seg[mask, :]
    #Fk0 = GT.activity[:, :T]

    print('Initializing variables...')

    Y = obs.data_in
    P0 = opts.P.T  # transpose

    #Nsk = groundtruth.seg.shape[1]
    #Sk = 1e-3 * sp.rand(Nvoxk, Nsk)  # +Sk0



    Nsk = Sk.shape[1]
    Nsu = Su.shape[1]


    print '#Sk:', Nsk, '#Su:', Nsu
    print 'Done initialization!'

    return (Y,Sk,Fk,Su,Fu,GT, P0, mask, masks)


# def solveOneFrame(frameDataIn, P0):  #framedata has structure [framenumber, y[:,framenumber]]
#     from scipy import optimize
#     #return frameDataIn[0]
#     Pt = Pu(P0,frameDataIn[0])
#     PSk = Pt.dot(Sk_bc.value).toarray()
#     PSu = Pt.dot(Su_bc.value)
#     PS = np.concatenate((PSk, PSu), axis=1)
#     F = optimize.nnls(PS,frameDataIn[1])
#     return F[0]

# def test1(frameDataIn, P0):
#     #     #from scipy import optimize
#     #     #Pt = Pu(P0,frameDataIn[0])
#     return frameDataIn[0]

def solveFgivenS(Y, Sk, Su, masks, P0, sc):
    frameData = [(i, Y[:,i]) for i in range(Y.shape[1])]

    Sk_bc = sc.broadcast(Sk) #sparkcontext must be created in the top level runexpt.py
    Su_bc = sc.broadcast(Su)

    def solveOneFrame(frameDataIn, P0):  #framedata has structure [framenumber, y[:,framenumber]]
        #from scipy import optimize
        #return frameDataIn[0]
        Pt = Pu(P0,frameDataIn[0])
        PSk = Pt.dot(Sk_bc.value).toarray()
        PSu = Pt.dot(Su_bc.value)
        PS = np.concatenate((PSk, PSu), axis=1)
        F = optimize.nnls(PS,frameDataIn[1])
        return F[0]

    F_solved = np.array(sc.parallelize(frameData,len(frameData)).map(lambda x: solveOneFrame(x, P0)).collect())

    Fk = F_solved[:, 0:Sk.shape[1]].T
    Fu = F_solved[:, Sk.shape[1]:(Sk.shape[1]+Su.shape[1])].T


    #NND filter the F timecourse
    Fk = applyNND(Fk)
    Fu = applyNND(Fu)
    return Fk,Fu


def applyNND(F, tau=5, tau_reverse=2):
    k = sp.exp(-(sp.arange(7*tau+1).astype('float')/tau))   #the convolution kernel
    #k_reverse = sp.exp(-(sp.arange(7*tau_reverse+1).astype('float')/tau_reverse))   #the convolution kernel
    print 'applying NND...'
    p = np.percentile(F,5,axis=1)
    for n in range(F.shape[0]):
        print n
        F[n,:] = np.convolve(k, NN_KP(F[n,:]-p[n], tau)[:,0], 'same') + p[n]
        #F[n, ::-1] = np.convolve(k_reverse, NN_KP(F[n,::-1]-p[n], tau_reverse)[:,0], 'same') + p[n]
    return F