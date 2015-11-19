
import numpy as np
import scipy as sp
from scipy import io
from numpy import ndarray
import pdb
import time
import random

fn = 'C:\Users\podgorskik\Documents\GitHub\PSI_simulations\PSI\PRECOMP_nonoise_py'

#def initialize(fn):
D = io.loadmat(fn, struct_as_record=False,squeeze_me = True)
#print type(D['ground_truth']['seg'])
#reconstruct_imaging_ADMM(D['obs'],D['opts'],1000,40,D['ground_truth'])
obs = D['obs']
opts = D['opts']
nIter = 1000
Nframes = 40
groundtruth = D['ground_truth']


#def reconstruct_imaging_ADMM(obs, opts, nIter, Nframes, groundtruth):

rho = 1e-0
lambdaFk_nuc = 1e-10
lambdaFu_nuc = 1e0
lambdaFk_TF = 1e-10
lambdaFu_TF = 1e-10

bw = groundtruth.bw


mask = ndarray.flatten(bw)>0
Nvoxk = sum(mask)

print(opts.P.shape)
obs.data_in = opts.P[mask,:].T.dot(groundtruth.seg[mask,:]).dot(groundtruth.activity)


[Nvox, Np] = opts.P.shape
[Np, T] = obs.data_in.shape
Nsk = groundtruth.activity.shape[0]
Nsu = groundtruth.unsuspectedPos.shape[1]

Sk0 = groundtruth.seg[mask, :]
Fk0 = groundtruth.activity[:, :T]

print('Initializing variables...')
tic = time.time()

Y = obs.data_in
P0 = opts.P.T  #transpose

X = sp.zeros((Np,T))
Xhat = sp.zeros((Np,T))
Xhat_nn = sp.zeros((Np,T))


#TODO: improve initialization!
Sk = 1e-3*sp.rand(Nvoxk,Nsk)   #+Sk0
Fk = 5e-1*sp.rand(Nsk,T)      #+Fk0

Su = 1e-3*sp.rand(Nvox,Nsu)
Fu = 5e-2*sp.rand(Nsu,T)

Fu_nuc = Fu    #zeros(size(Fu));
Fu_nn = Fu     #zeros(size(Fu));
Fu_TF = Fu     #zeros(size(Fu));
Fu_TF_z = Fu   #zeros(size(Fu));
Fu_TF_u = Fu   #zeros(size(Fu));
Fk_nuc = Fk    #zeros(size(Fk));
Fk_nn = Fk     #zeros(size(Fk));
Fk_TF = Fk     #zeros(size(Fk));
Fk_TF_z = Fk   #zeros(size(Fk));
Fk_TF_u = Fk   #zeros(size(Fk));


U_X = 0*sp.ones((Np,T))
U_Fu_nuc = 0*sp.ones(Fu.shape)
U_Fu_TF = 0*sp.ones(Fu.shape)
U_Fu_nn = 0*sp.ones(Fu.shape)
U_Fk_nuc = 0*sp.ones(Fk.shape)
U_Fk_TF = 0*sp.ones(Fk.shape)
U_Fk_nn = 0*sp.ones(Fk.shape)
U_Su_nn = 0*sp.ones(Su.shape)
U_Sk_nn = 0*sp.ones(Sk.shape)

print 'Done initialization!'

if Nframes < T:
    tidx = random.sample(range(1,T+1),Nframes)
else:
    tidx = np.arange(1,T+1)  #1:T

tidx2 = np.arange(1,T+1)

print(mask.shape)
#print(mask[0])







print time.time()-tic, 'sec Elapsed'

if __name__ == '__main__':
    initialize(binLocation)
