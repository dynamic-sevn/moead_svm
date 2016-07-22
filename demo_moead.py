# -*- coding: utf-8 -*-
"""
Author: Xi Lin <xi.lin@my.cityu.edu.hk> 
website: http://www.cs.cityu.edu.hk/~xilin4/
github:

This code is a demo for this paper:

    A Decomposition based Multiobjective Evolutionary Algorithm 
    with Classification 
    Xi Lin, Qingfu Zhang, Sam Kwong 
    Proceedings of the 2016 IEEE Congress on Evolutionary Computation (CEC16),
    Vancouver, Canada, July 2016 

"""

import os
import sys
import timeit

import numpy as np
from func_moead import *

path = os.path.abspath(os.path.dirname(sys.argv[0]))
start = timeit.default_timer()
evalcounter = 0

#name of test instance
ins = 'UF1'  

#initialize parameters and subproblems for moea/d
dim, nobj, popsize, niche, stop_nfeval = init_params(ins)
params = Params(popsize,niche,'tc',2500,stop_nfeval,0.9,2,0.5,1)
mop = Problem(ins,dim,nobj)
subproblems, idealpoint = init_subproblem(mop,params)

#load "true" pf to calculate IGD value
truepf = np.loadtxt(path + "/PF/%s.dat"%mop.name)

while not terminate(evalcounter,params):
    
    for i in range(params.popsize):
        
        #decide on whether choosing parents from the neighbourhood or not
        updateneighbour = np.random.rand(1)[0] < params.updateprob
        
        #generate new solution
        newpoint = genetic_op(i,updateneighbour,mop,
                              params,subproblems,'current')
                              
        #evaluate new solution
        newpoint.value = mop.evaluate(newpoint.parameter)
        evalcounter = evalcounter + 1
        
        #update estimated idealpoint
        idealpoint = np.minimum(idealpoint,newpoint.value)
        
        #update current population
        update_vec(i,updateneighbour,newpoint,mop,
                   params,subproblems,idealpoint)
        
        #calculate and display igd value
        if evalcounter%3000 == 0:
            print evalcounter
            pf = np.array(
                [subproblems[i].curpoint.value for i in range(params.popsize)])
                
            print calculateigd(truepf,pf) 
             
stop = timeit.default_timer()

#print running time
print stop - start     



