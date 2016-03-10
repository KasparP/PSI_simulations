import os, sys
#sys.path.append('/groups/turaga/home/turagas/research/PSI_simulations/Python/')
#sys.path.append('Y:/podgorskik/PSI_simulations/Python/')
sys.path.append('/tier2/turaga/podgorskik/PSI_simulations/Python')

import SLAPMi
from pyspark import SparkConf, SparkContext

os.environ['PYTHONPATH'] = (os.environ['PYTHONPATH'] + '/tier2/turaga/podgorskik/PSI_simulations/Python/SLAPMi/:')
print (os.getenv('PYTHONPATH'))

conf = SparkConf().setAppName('SLAPmi_experiment')
sc = SparkContext(conf=conf)
# sc.addPyFile('/tier2/turaga/podgorskik/PSI_simulations/Python/SLAPMi/slapmi.py')
# sc.addPyFile('/tier2/turaga/podgorskik/PSI_simulations/Python/SLAPMi/__init__.py')

(Y,Sk,Fk,Su,Fu,GT,P0,mask,masks) = SLAPMi.prepexpt('../Problem_nonoise_1Kframes_init.mat')
Sk,Fk,Su,Fu = SLAPMi.reconstruct_cpu(Y,Sk,Fk,Su,Fu,P0,mask,masks,Nframes=2,nIter=int(1e5),eta=1e-7,mu=0.9,adagrad=True, groundtruth=GT, sparkcontext=sc)

