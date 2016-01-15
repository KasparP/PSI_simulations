#import lsqlin
import numpy as np
import scipy as sp

def SolveFgivenS (Y, P0, S, motion = None):
    def P (frame):
        return P0

    #S is the concatenation [Sk Su]
    F = np.zeros((S.shape[1], Y.shape[1]))
    for frame in range(0,Y.shape[1]):
        PS = P(frame)*S
        print frame
        F[:,frame] = sp.optimize.nnls(PS,Y)
        #ret = lsqlin.lsqnonneg(PS, Y, {'show_progress': False})
        #F[:, frame] = ret['x']
    return F













