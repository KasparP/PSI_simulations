import os, sys
sys.path.append('/groups/turaga/home/turagas/research/PSI_simulations/Python/')
import SLAPMi

(Y,Sk,Fk,Su,Fu) = SLAPMi.prepexpt()
Sk,Fk,Su,Fu = SLAPMi.reconstruct_cpu(Y,Sk,Fk,Su,Fu,Nframes=2,nIter=int(1e5),eta=1e3,mu=0.9,adagrad=True)

