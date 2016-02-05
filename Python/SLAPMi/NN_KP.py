import scipy as sp
import numpy as np


def NN_KP (data, tau):
#performs nonnegative deconvolution with an exponential kernel
# returns counts: the deconvolved signal
    # data  :   The data to be deconvolved
    # tau   :   The time constant of the kernel in data samples

    T = len(data)
    counts = sp.zeros((T,1))
    counts[-1] = data[-1]
    cutoff = 8*tau             #how long the effect of a timepoint can be
    k = sp.exp(-(sp.arange(cutoff+1).astype('float')/tau))   #the convolution kernel
    recent = np.zeros((T,1)) * np.nan #stored locations where we assigned counts
    recent[0] = T-1
    recent_ix = 0



    #the points that could potentially be assigned counts:
    points = np.logical_and(data[:-1]>k[1]*np.hstack((0., data[:-2])), data[:-1]>0.)

    runstarts, = np.logical_and(points,~np.hstack((False, points[:-1]))).nonzero()
    runends, = np.logical_and(points, ~np.hstack((points[1:], False))).nonzero()
    run_id = len(runends)-1

    while run_id>=0:
        converged = False
        oldtop = 0
        oldbottom = 0
        t = runends[run_id]
        t1 = int(t)
        accum = 0

        while not converged:
            if recent_ix>0 and recent[recent_ix]<(t+cutoff):
                t2 = int(recent[recent_ix])
                C_max = counts[int(t2)]/k[int(t2-t)]
            else:
                t2 = int(sp.fmin(t+cutoff-1, T-1))
                C_max = float('inf')

            #b: kernel
            b = k[(t1-t):(t2-t)]
            top = b.dot(data[t1:t2])+oldtop
            bottom = sp.sum(np.square(b))+oldbottom
            done = False

            while not done:
                #the error function we are minimizing is sum((data-kernel.*C)^2)
                best_C = sp.fmax(top/bottom, 0)  #C=top/bottom sets the derivative of the error to 0

                if best_C>(C_max+accum):
                    if counts[t2]:
                        accum += counts[t2]/k[t2-t]
                        counts[t2] = 0
                    t1 = t2
                    oldtop = top
                    oldbottom = bottom
                    recent_ix -= 1

                    done = True

                else:

                    if t == runstarts[run_id] or data[t-1] < (best_C/k[1]):
                        if recent_ix>=0 and t2<=t+cutoff:
                            counts[t2] -= (best_C-accum)*k[t2-t]
                        run_start = runstarts[run_id]
                        init_ix = recent_ix + 1
                        recent_ix = recent_ix + 1 + t - run_start
                        recent[init_ix:recent_ix+1,0] = np.arange(t,run_start-1, -1)
                        counts[runstarts[run_id]:t+1, 0] = np.concatenate((data[run_start:t], best_C), axis=None) - np.concatenate((0,k[1]*data[run_start:t]), axis=None)

                        done = True
                        converged = True

                        #print('Keyboard 1:')
                        #code.interact(local=locals())
                    else:
                        t = t-1
                        runends[run_id] = t
                        accum = accum/k[1]
                        top = top*k[1] + data[t]
                        bottom = bottom*(k[1]**2) +1

                        #print('Keyboard 2:')
                        #code.interact(local=locals())

        run_id = run_id-1

    return counts


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import code                  #code.interact(local=locals()) is similar to matlab 'keyboard'
    import time

    tau = 100
    cutoff = 800
    A = 1+np.sin(np.arange(5000000).astype('float')/20000)
    print 'Example:'
    print 'Deconvolving ',  A.shape, 'timepoints'
    e = time.time()
    B = NN_KP(A, tau)
    print 'Seconds elapsed: ', time.time()-e
    plt.plot(A)
    plt.plot(B, 'r')

    #code.interact(local=locals())

    recon = np.convolve(sp.exp(-(sp.arange(cutoff).astype('float')/tau)), B[:,0])
    plt.plot(recon, 'g')
    plt.show()