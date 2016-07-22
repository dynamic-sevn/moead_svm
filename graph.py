# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 10:42:16 2016

@author: xilin4
"""


import numpy as np
import pickle
from matplotlib import pyplot as plt
from pylab import *


def plotIGD(ins_name,nfeval):
    
    with open('MOEAD.pickle') as f:
        results_MOEAD_ts,runtime_MOEAD_ts = pickle.load(f)    

    with open('MOEAD_svm.pickle') as f:
        results_MOEADBO_ts,runtime_MOEADBO_ts = pickle.load(f)   

    IGD_MOEAD_ts = np.median(results_MOEAD_ts[ins_name],axis = 0)
    IGD_MOEADBO_ts = np.median(results_MOEADBO_ts[ins_name],axis = 0)
    ymin = np.min([IGD_MOEAD_ts,IGD_MOEADBO_ts])
    ymax = np.max([IGD_MOEAD_ts,IGD_MOEADBO_ts])
    
    X = np.linspace(0, nfeval, 101, endpoint=True)
    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.plot(X,IGD_MOEAD_ts, color='blue', lw=2, label = 'MOEA/D')
    axes.plot(X,IGD_MOEADBO_ts, color='green', lw=2, label = 'MOEA/D-BO')
    
    axes.set_ylim([ymin,ymax])
    
    axes.set_yscale('log')
    axes.set_xlabel('# Function Evaluation')
    axes.set_ylabel('Log IGD')
    axes.legend(loc='upper right')
    axes.set_title(ins_name)    
    
    axes.grid(True)
    
    #return fig



def plotRate(ins_name,nfeval):
    
    with open('MOEAD_svm_rate.pickle') as f:
        results_MOEAD_svm_rate,results_pf_svm_rate,runtime_MOEAD_svm_rate,rate_all,rate_positive,rate_negative = pickle.load(f)            
    
    #plt.plot(range(list_len),rate_list_moead_svm_true,'b-',range(list_len),rate_list_moead_svm_raw,'g-')
    
    X = np.linspace(0, nfeval, 101, endpoint=True)
    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    y1 = np.median(rate_positive[ins_name],axis = 0)
    y2 = np.median(rate_negative[ins_name],axis = 0)
    
    axes.plot(X,y1, color='green', lw=2, label = 'positive')
    axes.plot(X,y2, color='blue', lw=2, label = 'negative')
    
    
    axes.set_xlabel('# Function Evaluation')
    axes.set_ylabel('Probability of Contributing')
    axes.legend(loc='upper right')
    axes.set_title(ins_name)    
    
    axes.grid(True)
    
    #return fig


def plotPF(ins_name,gen):
    
    with open('MOEAD_Rate.pickle') as f:
        results_MOEAD_rate,results_pf,runtime_MOEAD_rate,rate_all = pickle.load(f)           
    
    with open('MOEAD_svm_rate.pickle') as f:
       results_MOEAD_svm_rate,results_pf_svm,runtime_MOEAD_svm_rate,rate_all,rate_positive,rate_negative = pickle.load(f)            
    
    #plt.plot(range(list_len),rate_list_moead_svm_true,'b-',range(list_len),rate_list_moead_svm_raw,'g-')
    
    #scatter(results_pf_svm[ins_name][0][gen][:,0],results_pf_svm[ins_name][0][gen][:,1])
    truepf = np.loadtxt("D:\\MOEAD_python_classification\\PF\\%s.dat"%ins_name)

    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.scatter(results_pf[ins_name][0][gen][:,0],results_pf[ins_name][0][gen][:,1],
                 color='b', alpha=0.5,edgecolors='none',label = 'MOEA/D')
    axes.scatter(results_pf_svm[ins_name][0][gen][:,0],results_pf_svm[ins_name][0][gen][:,1],
                 color='r', alpha=0.5,edgecolors='none',label = 'MOEA/D-SVM')
    axes.plot(truepf[:,0],truepf[:,1],
                 color='k',lw=2, label = 'PF')
    
    
    #axes.set_xlim([0,3]) 
    
    #axes.set_ylim([0,3]) 
    
    axes.set_xlabel('f1')
    axes.set_ylabel('f2')
    axes.legend(loc='upper right')
    axes.set_title(ins_name)    
    
    axes.grid(True)
    
    #return fig


def plotIGDSensity(ins_name,nfeval):
    
    with open('MOEAD_100.pickle') as f:
        results_MOEAD,runtime_MOEAD = pickle.load(f)    
    
    with open('MOEAD_GPC_100.pickle') as f:
        results_MOEAD_GPC,runtime_MOEAD_GPC = pickle.load(f)        
    
    with open('MOEAD_SVM_Sensity2_100.pickle') as f:
        results_MOEAD_SVM_Sensity2, = pickle.load(f)              
    
    
    results = results_MOEAD_SVM_Sensity2[ins_name]
    value = np.power(10,np.arange(0,2.5,1))    
    
    
    X = np.linspace(0, nfeval, 101, endpoint=True)
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    
    median_IGD = np.median(results_MOEAD[ins_name],axis = 0)
    median_IGD_GPC = np.median(results_MOEAD_GPC[ins_name],axis = 0)
            
    ymin = np.min([median_IGD,median_IGD_GPC])
    ymax = np.max([median_IGD,median_IGD_GPC])
            
    axes.plot(X,median_IGD , 'r--',lw=2, label = 'MOEA/D')
    axes.plot(X,median_IGD_GPC , 'b.-',lw=2, label = 'MOEA/D-GPC')
    
    for g_index in range(len(value)):
        for c_index in range(len(value)):
            median_IGD = np.median(results[g_index,c_index],axis = 0)
            
            ymin = min(ymin,np.min(median_IGD))
            ymax = max(ymax,np.max(median_IGD))
            
            axes.plot(X,median_IGD , lw=2, label = r'$\gamma=%d,C=%d$'%(value[g_index],value[c_index]), alpha=0.3)
    
    axes.set_ylim([ymin,ymax])
    axes.set_yscale('log')
    axes.set_xlabel('# Function Evaluation')
    axes.set_ylabel('Log IGD')
    axes.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    axes.set_title(ins_name)    
    
    axes.grid(True)
    
    #return fig




ZDTfig = plotIGD('ZDT1',2000)
ZDTfig.savefig('ZDT1.png')

UFfig = plotIGD('UF1',2000)
UFfig.savefig('UF1.png')


#plotPF(shrinkpf(pf_random),shrinkpf(pf_moead),shrinkpf(pf_bo))
def plotPF(pf1,pf2,pf3):
    fig = plt.figure()

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
    
    #label_color_map = {0 : 'r', 1 : 'k', 2 : 'b',3 : 'g'}
    
    p1 = axes.scatter(-pf1[:,0],-pf1[:,1],color = 'blue')
    p2 = axes.scatter(-pf2[:,0],-pf2[:,1],color = 'red')
    p3 = axes.scatter(-pf3[:,0],-pf3[:,1],color = 'green')

    axes.set_xlabel('precision')
    axes.set_ylabel('recall')
    axes.grid(True)
    
    axes.legend((p1,p2,p3),
           ('random', 'MOEA/D', 'MOEA/D-BO'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)    
    
    return fig
    
svmfig = plotPF(shrinkpf(pf_random),shrinkpf(pf_moead),shrinkpf(pf_bo))
svmfig.savefig('svmfig.png')
    
    
    
    
    