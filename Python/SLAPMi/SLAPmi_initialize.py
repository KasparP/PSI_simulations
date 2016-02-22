# run using:
# spark-janelia -n7 lsd -s SLAPmi_initialize.py [fullpath]
# runs with 7 nodes
from pyspark import SparkConf, SparkContext
import sys
import numpy as np
import scipy as sp
from scipy import io
from scipy import optimize
#import code


def SLAPmi_initialize_spark(fullpath):
    D = io.loadmat(fullpath, struct_as_record=False, squeeze_me=True)

    obs = D['obs']
    opts = D['opts']

    Y = obs.data_in
    P0 = opts.P.T  # transpose

    Sk = D['Sk']
    Su = D['Su']
    if len(Su.shape)<2:
        Su = Su[:,None]

    masks = D['masks']
    #S = Sk
    #S = np.concatenate((Sk,Su), axis=1)

    def P (frame):
        return P0


    def solveOneFrame(frameDataIn):  #framedata has structure [framenumber, y[:,framenumber]]
        Pt = P(frameDataIn[0])
        #PSk = np.zeros((Pt.shape[0], Sk.shape[0]))
        #for Sk_ix in range(len(Sk)):
        #    PSk[:, Sk_ix] = Pt[:,masks[:,Sk_ix].toarray()[:,0]].dot(Sk[Sk_ix])
        #code.interact(local=locals())
        PSk = Pt.dot(Sk_bc.value).toarray()
        PSu = Pt.dot(Su_bc.value)
        PS = np.concatenate((PSk, PSu), axis=1)
        F = optimize.nnls(PS,frameDataIn[1])

        #code.interact(local=locals())

        return F[0]


    #code.interact(local=locals())

    conf = SparkConf().setAppName('SLAPmi_initialize')
    sc = SparkContext(conf=conf)

    Sk_bc = sc.broadcast(Sk)
    Su_bc = sc.broadcast(Su)

    frameData = [(i, Y[:,i]) for i in range(Y.shape[1])]

    F_solved = np.array(sc.parallelize(frameData,len(frameData)).map(solveOneFrame).collect())

    #
    print 'F_solved', F_solved.shape
    print 'Sk', Sk.shape
    print 'Su', Su.shape

    Fk = F_solved[:, 0:Sk.shape[1]].T
    Fu = F_solved[:, Sk.shape[1]:(Sk.shape[1]+Su.shape[1])].T

    return Sk,Su,Fk,Fu, obs, opts, masks, D['ground_truth']

if __name__ == '__main__':
    fullpath = sys.argv[1]
    [Sk,Su,Fk,Fu, obs, opts, masks, GT] = SLAPmi_initialize_spark(fullpath)
    io.savemat(fullpath[:-4] + '_init.mat',{'obs': obs, 'opts':opts, 'GT': GT, 'Sk':Sk, 'Su':Su,'Fk':Fk,'Fu':Fu, 'masks':masks})