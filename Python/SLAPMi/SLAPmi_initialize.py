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

    #S = Sk
    #S = np.concatenate((Sk,Su), axis=1)

    def P (frame):
        return P0

    def solveOneFrame(frameDataIn):  #framedata has structure [framenumber, y[:,framenumber]]
        Pt = P(frameDataIn[0])
        PS = np.concatenate((Pt*Sk_bc.value.toarray(), Pt*Su_bc.value), axis=1)
        F = optimize.nnls(PS,frameDataIn[1])
        return F[0]


    #code.interact(local=locals())

    conf = SparkConf().setAppName('SLAPmi_initialize')
    sc = SparkContext(conf=conf)

    Sk_bc = sc.broadcast(Sk)
    Su_bc = sc.broadcast(Su)

    frameData = [(i, Y[:,i]) for i in range(Y.shape[1])]

    #frameData = [(i, Y[:,i]) for i in range(5)]

    F_solved = np.array(sc.parallelize(frameData,len(frameData)).map(solveOneFrame).collect())
    #F_solved = sc.parallelize(frameData).map(lambda x: solveOneFrame(x)).collect()

    #
    print 'F_solved', F_solved.shape
    print 'Sk', Sk.shape
    print 'Su', Su.shape

    Fk = F_solved[:, 0:Sk.shape[1]]
    Fu = F_solved[:, Sk.shape[1]:(Sk.shape[1]+Su.shape[1])]

    #Fk = F_solved[:, range(Sk.shape[0])]
    #Fu = F_solved[:, Sk.shape[0]+range(Su.shape[0])]

    return Sk,Su,Fk,Fu, obs, opts, D['ground_truth']

if __name__ == '__main__':
    fullpath = sys.argv[1]
    [Sk,Su,Fk,Fu, obs, opts, GT] = SLAPmi_initialize_spark(fullpath)
    io.savemat(fullpath[:-4] + '_init.mat',{'obs': obs, 'opts':opts, 'GT': GT, 'Sk':Sk, 'Su':Su,'Fk':Fk,'Fu':Fu})
