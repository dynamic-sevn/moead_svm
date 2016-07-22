# -*- coding: utf-8 -*-

"""
Author: Xi Lin <xi.lin@my.cityu.edu.hk> 
website: http://www.cs.cityu.edu.hk/~xilin4/
github:

This code is a demo for this paper:

    A Decomposition based Multiobjective Evolutionary Algorithm 
    with Classification 
    Xi Lin, Qingfu Zhang, Sam Kwong 
    IEEE World Congress on Computational Intelligence(IEEE WCCI), 
    Vancouver, Canada, July 2016 

"""

import os
import sys
import timeit

import numpy as np
import pickle

from func_moead import *

path = os.path.abspath(os.path.dirname(sys.argv[0]))

ins = ['ZDT1','ZDT2','ZDT3','ZDT4','ZDT6','DTLZ1','DTLZ2',
'UF1','UF2','UF3','UF4','UF5','UF6','UF7','UF8','UF9','UF10']

num_run = 30

results_MOEAD = {}
runtime_MOEAD = np.full([len(ins),num_run],np.nan)

for k in range(len(ins)):

    dim, nobj, popsize, niche, stop_nfeval = init_params(ins[k])
    params = Params(popsize,niche,'tc',2500,stop_nfeval,0.9,2,0.5,1)
    
    mop = Problem(ins[k],dim,nobj)
    
    results_MOEAD[ins[k]] = np.full([num_run,101],np.nan)
    truepf = np.loadtxt(path + "/PF/%s.dat"%mop.name)
   
    for j in range(num_run):
        start = timeit.default_timer()
        evalcounter = 0
       
        subproblems, idealpoint = init_subproblem_classification(mop,params)
        
        pf = np.array([subproblems[i].curpoint.value for i in range(params.popsize)])
        igd = calculateigd(truepf,pf) 
        results_MOEAD[ins[k]][j,0] = igd
        
        while not terminate(evalcounter,params):
            
            #train SVM model
            classifier = trainSVMmodel(subproblems,params)
    
            for i in range(params.popsize):
                updateneighbour = np.random.rand(1)[0] < params.updateprob
                
                newpoint = genetic_op(i,updateneighbour,mop,params,subproblems,'current')
                
                label = classifier.predict(newpoint.parameter.reshape(1,-1))
                r = np.random.rand(1)[0]
        
                if label == 1 or r <= 0.1:
                    newpoint.value = mop.evaluate(newpoint.parameter)
                    evalcounter = evalcounter + 1
           
                    idealpoint = np.minimum(idealpoint,newpoint.value)
                    update_vec(i,updateneighbour,newpoint,mop,params,subproblems,idealpoint)
                    
                    size = stop_nfeval/100
                    
                    if evalcounter%size == 0:
                        pf = np.array([subproblems[i].curpoint.value for i in range(params.popsize)])
                        igd = calculateigd(truepf,pf) 
                        results_MOEAD[ins[k]][j,evalcounter/size] = igd
                        print evalcounter
                        print igd
                        
                    if terminate(evalcounter,params):
                        break
                    
        stop = timeit.default_timer()

        runtime_MOEAD[k,j] =  stop - start     
        print ins[k],j
        print runtime_MOEAD[k,j]
        

#save resutls
with open('MOEAD_svm.pickle', 'w') as f:
    pickle.dump([results_MOEAD,runtime_MOEAD], f)

#load resutls
with open('MOEAD_svm.pickle') as f:
    results_MOEAD_svm,runtime_MOEAD_svm = pickle.load(f)        