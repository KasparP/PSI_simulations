import os, sys
sys.path.append('/groups/turaga/home/turagas/research/PSI_simulations/Python/')
import SLAPMi

(Y,Sk,Fk,Su,Fu) = SLAPMi.prepexpt()
Sk,Fk,Su,Fu = SLAPMi.reconstruct_cpu(Y[:,:25],Sk,Fk[:,:25],Su,Fu[:,:25],Nframes=200,nIter=int(1e5),eta=1e1,mu=0.7,adagrad=True)

