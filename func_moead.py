# -*- coding: utf-8 -*-
"""
Author: Xi Lin <xi.lin@my.cityu.edu.hk> 
website: http://www.cs.cityu.edu.hk/~xilin4/
github:

This code is a demo for this paper:

    A Decomposition based Multiobjective Evolutionary Algorithm 
    with Classification 
    Xi Lin, Qingfu Zhang, Sam Kwong 
    Proceedings of the 2016 IEEE Congress on Evolutionary Computation (CEC16)
    Vancouver, Canada, July 2016 

"""
import os
import sys
import copy

import numpy as np
import scipy
from sklearn import svm

from test_instances import *

path = os.path.abspath(os.path.dirname(sys.argv[0]))


class Problem:
    """Multi-objective Problem
    
    name: MoP name
    dim: dimension of decision space
    nobj：dimension of objective space
    domain: domain for decision variable
    
    """
    def __init__(self,name,dim,nobj,domain = None):
        self.name = name
        self.dim = dim
        self.nobj = nobj
        if domain is None:
            self.domain = np.tile(np.array([0,1]),(self.dim,1))   
    
    def evaluate(self,x):
        #evaluate objective values for given decision variable
        return testfunc(self,x)
        

                
class Params:
    """Parameters for MOEA/D
    
    popsize: population size
    niche: neighbourhood size
    dmethod: decomposition method
    iteration: number of maximum iterations in each run; not used in this demo
    stop_nfeval: number of maximum function evaluations in each run
    updatedprob: probability that parent solutions are selected from the 
                neighbourhood, but not the whole population
    updatednb: maximum number of current solutions which would be replaced by 
                each new solution
    F, CR: parameters for DE operator
    
    """
    def __init__(self,popsize,niche,dmethod,iteration,
                 stop_nfeval,updateprob,updatenb,F,CR):
        self.popsize = popsize         
        self.niche = niche
        self.dmethod = dmethod
        self.iteration = iteration
        self.stop_nfeval = stop_nfeval
        
        self.updateprob = updateprob
        self.updatenb = updatenb
        
        self.F = F
        self.CR = CR
        

class Subproblem:
    """Subproblem in MOEA/D
    
    weight: decomposition weight
    neighbour: index of neighbours
    
    curpoint: Individual Class
              current best solution
              
    subpoint: Individual Class
              current sub-best solution, for classification training
    
    """
    def __init__(self,weight,mop,params):
        self.weight = weight
        self.neighbour = np.full(params.niche,np.nan)
        
        self.curpoint = np.full(mop.dim,np.nan)
        self.subpoint = np.full(mop.dim,np.nan)

class Individual:
    """Solution in MOEA/D
    
    parameter: decision variable
    value: objective value
    
    """
    def __init__(self,parameter):
        self.parameter = parameter
        self.value = float('Inf')
        

def init_params(ins):
    """Initialize parameters for test instance
    
    The given parameters in this function are the same as in the paper:
    
    A Decomposition based Multiobjective Evolutionary Algorithm with 
    Classification 
    Xi Lin, Qingfu Zhang, Sam Kwong 
    IEEE World Congress on Computational Intelligence(IEEE WCCI), 
    Vancouver, Canada, July 2016 
    
    Parameters
    ----------
    ins: name for test instance
    
    Returns
    -------
    dim: dimension of decision space
    nobj: dimenson of objective space
    popsize: population size
    niche: neighbourhood size
    stop_nfeval: number of maximum function evaluations in each run
    """
    
    if ins in ['ZDT1','ZDT2','ZDT3']:
        dim = 30
        nobj = 2
        popsize = 300
        niche = 30
        stop_nfeval = 100000
        
    if ins in ['ZDT4','ZDT6']:
        dim = 10
        nobj = 2
        popsize = 300
        niche = 30
        stop_nfeval = 100000
    
    if ins in ['DTLZ1']:
        dim = 10
        nobj = 3
        popsize = 595
        niche = 50
        stop_nfeval = 100000    
    
    if ins in ['DTLZ2']:
        dim = 30
        nobj = 3
        popsize = 595
        niche = 50
        stop_nfeval = 100000    
    
    if ins in ['UF1','UF2','UF3','UF4','UF5','UF6','UF7']:
        dim = 30
        nobj = 2
        popsize = 600
        niche = 30
        stop_nfeval = 300000

    if ins in ['UF8','UF9','UF10']:
        dim = 30
        nobj = 3
        popsize = 595
        niche = 50
        stop_nfeval = 300000

    return dim, nobj, popsize, niche, stop_nfeval

def init_point(mop):
    """Initialize a solution with randomly generated decision variable
        
    Parameters
    ----------
    mop: Problem Class
         multi-objective problem to be sloved
   
    
    Returns
    -------
    point: Individual Class
           a solution with randomly generated decision variable, which is not
           evaluated yet
    """
    lowend = mop.domain[:,0]
    span = mop.domain[:,1] - lowend
    
    para = lowend + span * np.random.rand(mop.dim)
    point = Individual(para)
    
    return point

def init_subproblem_classification(mop,params):
    """Initialize all subproblems and ideal point for MOEA/D-SVM
        
    Parameters
    ----------
    mop: Problem Class
         multi-objective problem to be sloved
    params: Params Class
            parameters for moea/d
    
    Returns
    -------
    subproblems: Subproblem Class
                 all subproblems initialized accroding to mop and params
    idealpoint: estimated idealpoint for Tchebycheff decomposition
    """
    
    #load already genereted weights vector in weight file
    weights = np.loadtxt(path 
                        + "/weight/W%dD_%d.dat"%(mop.nobj,params.popsize)) 
    idealpoint = np.ones(mop.nobj) * float('Inf')
    subproblems = []
    
    #initialize Subproblem Class for each weight vetor
    for i in range(params.popsize):
        sub = Subproblem(weights[i],mop,params)
        subproblems.append(sub)
    
    #distmat[i,j] is the distance btw sub[i] and sub[j], distmat[i,i] = nan
    distmat = np.full([params.popsize, params.popsize],np.nan)
    
    #initialize current best/sub-best point for each subproblem and idealpoint
    for i in range(params.popsize):
        for j in range(i+1,params.popsize):
            a = subproblems[i].weight
            b = subproblems[j].weight
            distmat[i,j] = np.linalg.norm(a - b)
            distmat[j,i] = distmat[i,j]
        
        #calculate the neighbourhood for each subproblem
        subproblems[i].neighbour = distmat[i,].argsort()[0:params.niche]
        
        subproblems[i].curpoint = init_point(mop)
        subproblems[i].curpoint.value = mop.evaluate(
                                            subproblems[i].curpoint.parameter)
        
        
        subproblems[i].subpoint = init_point(mop)
        subproblems[i].subpoint.value = mop.evaluate(
                                            subproblems[i].subpoint.parameter)
         

        idealpoint = np.minimum.reduce([idealpoint,
                                        subproblems[i].curpoint.value,
                                        subproblems[i].subpoint.value])
    #swap(curpoint,subpoint) if g_i(subpoint) < g_i(curpoint)
    #where g_i() is value function for the i-th subproblem
    for i in range(params.popsize):
        curvalue = subobjective_vec(subproblems[i].weight,
                                subproblems[i].curpoint.value.reshape(1,-1),
                                idealpoint,params.dmethod)
                                
        subvalue = subobjective_vec(subproblems[i].weight,
                                subproblems[i].subpoint.value.reshape(1,-1),
                                idealpoint,params.dmethod)

        if subvalue < curvalue:
            subproblems[i].curpoint, subproblems[i].subpoint = subproblems[i].subpoint, subproblems[i].curpoint
        
    return (subproblems, idealpoint)



def init_subproblem(mop,params):
    """Initialize all subproblems and ideal point for MOEA/D
        
    Parameters
    ----------
    mop: Problem Class
         multi-objective problem to be sloved
    params: Params Class
            parameters for moea/d
    
    Returns
    -------
    subproblems: Subproblem Class
                 all subproblems initialized accroding to mop and params
    idealpoint: estimated idealpoint for Tchebycheff decomposition
    """
    
    weights = np.loadtxt(path 
                        + "/weight/W%dD_%d.dat"%(mop.nobj,params.popsize)) 
    idealpoint = np.ones(mop.nobj) * float('Inf')
    subproblems = []
    
    #initialize Subproblem Class for each weight vetor
    for i in range(params.popsize):
        sub = Subproblem(weights[i],mop,params)
        subproblems.append(sub)
    
    #distmat[i,j] is the distance btw sub[i] and sub[j], distmat[i,i] = nan
    distmat = np.full([params.popsize, params.popsize],np.nan)
    
    #initialize current best/sub-best point for each subproblem and idealpoint
    for i in range(params.popsize):
        for j in range(i+1,params.popsize):
            a = subproblems[i].weight
            b = subproblems[j].weight
            distmat[i,j] = np.linalg.norm(a - b)
            distmat[j,i] = distmat[i,j]
        
        subproblems[i].neighbour = distmat[i,].argsort()[0:params.niche]
        
        subproblems[i].curpoint = init_point(mop)
        subproblems[i].curpoint.value = mop.evaluate(
                                            subproblems[i].curpoint.parameter)
        
        idealpoint = np.minimum(idealpoint,subproblems[i].curpoint.value)
    
    return (subproblems, idealpoint)


def terminate(n,params):
    """Decide on whether to terminate current algo run or not 
        
    Parameters
    ----------
    n: number of total evaluations have been conducted in current run
    params: Params Class
            parameters for moea/d
    
    Returns
    -------
    boolean expression
    True if number of total evaluations exceed params.stop_nfeval
    """
    return n >= params.stop_nfeval

def genetic_op(index,updateneighbour,mop,
               params,subproblems,ptype): 
    """Generated a new solutions for the index-th subproblem
        
    Parameters
    ----------
    index: subproblem index
    updateneighbour: boolean expression
                     whether parent solutions are selected from the 
                     neighbourhood or not
    mop: Problem Class
         multi-objective problem to be sloved  
    params: Params Class
            parameters for moea/d
    subproblems: Subproblem Class
                 all subproblems 
    ptype: the type of generated solutions, always "current" in this demo
    
    Returns
    -------
    newpoint: Individual Class
              a new generated solution
    
    """
    
    #select parents
    parents_index = mate_select(index,updateneighbour,
                                subproblems,params,2)
                                
    #generate a new solution using DE crossover
    newpoint = de_crossover(index,parents_index,subproblems,
                            params.F,params.CR,mop,ptype)
    #mutate new solution 
    mutate(newpoint,mop,1.0/mop.dim,20)
    return newpoint

def mate_select(index,updateneighbour,
                  subproblems,params,size):
    """Select parents for new solution generation
        
    Parameters
    ----------
    index: subproblem index
    updateneighbour: boolean expression
                     whether parents are selected from the neighbourhood 
                     or not
    subproblems: Subproblem Class
                 all subproblems 
    params: Params Class
            parameters for moea/d
    size: number of parents selected
    
    Returns
    -------
    selected_list: List, len(List) = size
              list of selected parents' indexes 
    
    """
    selected_list = []
    
    #decide on whether parents are selected from the neighbourhood or not
    if(updateneighbour):
        selindex = subproblems[index].neighbour
    else:
        selindex = range(params.popsize)
    
    #select list of selected parents' indexes
    while len(selected_list) < size:
        r =  np.random.rand(1)[0]
        parent = selindex[np.int(np.floor(len(selindex)*r))]
        
        if (not parent in selected_list):
            selected_list.append(parent)
    return selected_list
    
def de_crossover(index,parents_index,subproblems,F,CR,mop,ptype):
    """Generate a new solution using DE crossover
        
    Parameters
    ----------
    index: subproblem index
    parents_index: List
                    list of selected parents' indexes 
    subproblems: Subproblem Class
                 all subproblems 
    F,CR: DE parameters
    mop: Problem Class
         multi-objective problem to be sloved  
    ptype: the type of generated solutions, always "current" in this demo
    
    Returns
    -------
    newpoint: Individual Class
              a new generated solution
    
    """
    
    #initialize new solution with randomly generated decision variable
    newpoint = init_point(mop)
    
    #decide the decision variable using DE crossover
    if ptype == 'current':
        x1 = subproblems[index].curpoint.parameter
        x2 = subproblems[parents_index[0]].curpoint.parameter
        x3 = subproblems[parents_index[1]].curpoint.parameter
        cross = x1 + F * (x2 - x3)
        
        newpoint.parameter = np.copy(subproblems[index].curpoint.parameter)
    
    crossindex = np.random.rand(mop.dim) < CR
    newpoint.parameter[crossindex] = cross[crossindex]
    
    for i in range(mop.dim):
        r1 =  np.random.rand(1)[0]
        if r1 < CR:
            newpoint.parameter[i] = cross[i]
        
        #handle the boundary
        lowerbound = mop.domain[i,0]
        upperbound = mop.domain[i,1]
        
        if newpoint.parameter[i] < lowerbound:
            r2 =  np.random.rand(1)[0]
            newpoint.parameter[i] = lowerbound + r2*(x1[i] - lowerbound)
        if newpoint.parameter[i] > upperbound:
            r2 =  np.random.rand(1)[0]
            newpoint.parameter[i] = upperbound - r2*(upperbound - x1[i])
               
    return newpoint

def mutate(newpoint,mop,rate,eta):
    """Mutate new generated solution
        
    Parameters
    ----------
    index: subproblem index
    mop: Problem Class
         multi-objective problem to be sloved  
    rate,eta: mutation parameters
    
    Returns
    -------
    newpoint is mutable, hence no return is needed
    
    """
    
    #polynomial mutate
    for i in range(mop.dim):
        r1 = np.random.rand(1)[0]
        
        if(r1 < rate):
            y = newpoint.parameter[i]
            yl = mop.domain[i,0]
            yu = mop.domain[i,1]
            
            r2 = np.random.rand(1)[0]
            
            if(r2 < 0.5):
                sigma = (2 * r2) ** (1.0/(eta + 1)) - 1
            else:
                sigma = 1 - (2 - 2*r2) ** (1.0/(eta + 1))
            
            newpoint.parameter[i] = y + sigma * (yu - yl)
         
            if newpoint.parameter[i] > yu:
                newpoint.parameter[i] = yu
            if newpoint.parameter[i] < yl:
                newpoint.parameter[i] = yl
                     


def subobjective_vec(weight,value,idealpoint,dmethod):
    """Calculate the value of subproblem with given weight, 
                value and idealpoint
        
    Parameters
    ----------
    weight: weight vector
    value: objective value 
    idealpoint: idealpoint
    dmethod: decomposition method; in this demo, dmethod is always 'tc' which
             stands for Tchebycheff decomposition
    
    Returns
    -------
    mutated_newpoint: Individual Class
                      a new generated solution
    
    """

    if dmethod is 'tc':
        new_weight = np.copy(weight)
        new_weight[new_weight == 0.0] = 0.0001
        absdiff = np.abs(value - idealpoint)
        return np.amax(new_weight * absdiff,axis = 1)


def update_vec(index,updateneighbour,newpoint,
               mop,params,subproblems,idealpoint):
    """Updated current population using new generated solutions
        
    Parameters
    ----------
    index: subproblem index
    updateneighbour: boolean expression
                     whether parent solutions are selected from the 
                     neighbourhood or not
    newpoint: Individual Class
              a new generated solution
    mop: Problem Class
         multi-objective problem to be sloved  
    params: Params Class
            parameters for moea/d
    subproblems: Subproblem Class
                 all subproblems 
    idealpoint: estimated idealpoint
    
    Returns
    -------
    is_updated: 1 if at least one current solution was replaced
                0 otherwise
    
    """
    is_updated = 0
    
    #Classes subproblems[k] k = 1,2,..., is mutable, hence no return is needed
    if(updateneighbour):
        updateindex = np.array(subproblems[index].neighbour)
    else:
        updateindex = np.array(range(params.popsize))
        
        
    np.random.shuffle(updateindex)   
    
    weight_vec = np.array([subproblems[k].weight for k in updateindex]) 
    oldvalue_vec = np.array(
                        [subproblems[k].curpoint.value for k in updateindex])
            
    
    oldobj_vec = subobjective_vec(weight_vec,oldvalue_vec,
                                  idealpoint,params.dmethod)
                                  
    newobj_vec = subobjective_vec(weight_vec,newpoint.value,
                                  idealpoint,params.dmethod)
    
    #contain maximum(not always) 2 elementc
    replaceindex = updateindex[newobj_vec < oldobj_vec][:2]

    for k in replaceindex:
        subproblems[k].subpoint = subproblems[k].curpoint
        subproblems[k].curpoint = newpoint
        is_updated = 1
        
    return is_updated
             
def calculateigd(truepf, pf):
    """Calculate IGD value for truepf and pf
        
    Parameters
    ----------
    truepf: "true" pf value 
    pf: estimated pf value
    
    Returns
    -------
    igd: IGD value
    
    """
    Y = scipy.spatial.distance.cdist(truepf, pf, 'euclidean')
    mindist = np.min(Y, axis=1)
    igd = np.mean(mindist)         
    return igd    

def trainSVMmodel(subproblems,params,gamma=1,C=100):
    """Train SVM Classification model
        
    Parameters
    ----------
    subproblems: Subproblem Class
                 all subproblems 
    params: Params Class
            parameters for moea/d
    gamma, C: kernel parameters
              In this demo, the kernel we use is always RBF kernel with 
              fixed gamma = 1 and C = 100.
    
    Returns
    -------
    classifier: trained SVM classifier
    
    """
    
    #curpoints as positive samples, and subpoints as negative samples
    curX = np.array([subproblems[k].curpoint.parameter for k in range(params.popsize)])
    subX = np.array([subproblems[k].subpoint.parameter for k in range(params.popsize)])

    trainX = np.concatenate((curX, subX))
    trainLabel = np.concatenate((np.ones(params.popsize),
                                 np.zeros(params.popsize)))
    
    classifier = svm.SVC(gamma = gamma, C = C)
    classifier.fit(trainX, trainLabel)
    
    return classifier
        

        
        
        
        
        
        
        
        
        
    
    
    