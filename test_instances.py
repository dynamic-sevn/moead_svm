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

import numpy as np


class switch(object):
    #http://code.activestate.com/recipes/410692/
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
            
def testfunc(mop,x):
    f = np.zeros(mop.nobj)
        
    for case in switch(mop.name):
        """ZDT Test Instances"""
        if case('ZDT1'):
            s = sum(x[i] for i in range(1,mop.dim))
            g = 1.0 + 9.0 * s / (mop.dim - 1)
            f[0] = x[0]
            f[1] = g * (1.0 - np.sqrt(f[0]/g))
            return f
            
        if case('ZDT2'):
            s = sum(x[i] for i in range(1,mop.dim))
            g = 1.0 + 9.0 * s / (mop.dim - 1)
            f[0] = x[0]
            f[1] = g * (1.0 - np.power(f[0]/g,2))
            return f
            
        if case('ZDT3'):
            s = sum(x[i] for i in range(1,mop.dim))
            g = 1.0 + 9.0 * s / (mop.dim - 1)                
            f[0] = x[0]
            f[1] = g * (1.0 - np.sqrt(f[0]/g) - f[0]*np.sin(10 * np.pi * f[0])/g)
            return f    
            
        if case('ZDT4'):
            s = sum(np.power(10*(x[i] - 0.5),2)-10*np.cos(4*np.pi*10*(x[i] - 0.5)) for i in range(1,mop.dim))
            g = 1.0 + 10.0 *(mop.dim - 1) + s           
            f[0] = x[0]
            f[1] = g * (1.0 - np.sqrt(f[0]/g))
            return f             
            
        if case('ZDT6'):
            s = sum(x[i] for i in range(1,mop.dim))
            s = s/(mop.dim - 1.0)
            g = 1.0 + 9.0 * np.power(s,0.25)              
            f[0] = 1 - np.exp(-4*x[0])*np.power(np.sin(6*np.pi*x[0]),6)
            f[1] = g * (1.0 - np.power(f[0]/g,2)) 
            return f                
            
        """DTLZ Test Instances"""   
        if case('DTLZ1'):
            s = sum(np.power(x[i]-0.5,2) - np.cos(20*np.pi*(x[i] - 0.5)) for i in range(2,mop.dim))
            g = 100 * (mop.dim - 2 + s)
            f[0] = (1 + g)*x[0]*x[1]
            f[1] = (1 + g)*x[0]*(1 - x[1])
            f[2] = (1 + g)*(1 - x[0])
            return f
        
        if case('DTLZ2'):
            s = sum(np.power(2*(x[i] - 0.5),2) for i in range(2,mop.dim))
            g = s
            f[0] = (1 + g) * np.cos(x[0]*np.pi/2)*np.cos(x[1]*np.pi/2)
            f[1] = (1 + g) * np.cos(x[0]*np.pi/2)*np.sin(x[1]*np.pi/2)
            f[2] = (1 + g) * np.sin(x[0]*np.pi/2)
            return f
    
        """UF Test Instances"""
        if case('UF1'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
                yi = xi - np.sin(6.0*np.pi*x[0] + i*np.pi/mop.dim)
                yi = yi * yi
                
                if i % 2 == 0:
                    s2 = s2 + yi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi
                    count1 = count1 + 1.0

            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f
            
        if case('UF2'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
               
                if i % 2 == 0:
                    yi = xi-0.3*x[0]*(x[0]*np.cos(24.0*np.pi*x[0]+4.0*i*np.pi/mop.dim)+2.0)*np.sin(6.0*np.pi*x[0]+i*np.pi/mop.dim)
                    s2 = s2 + yi*yi
                    count2 = count2 + 1.0
                else:
                    yi = xi-0.3*x[0]*(x[0]*np.cos(24.0*np.pi*x[0]+4.0*i*np.pi/mop.dim)+2.0)*np.cos(6.0*np.pi*x[0]+i*np.pi/mop.dim)
                    s1 = s1 + yi*yi
                    count1 = count1 + 1.0
                    
            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f

        if case('UF3'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            prod1 = prod2 = 1.0
            
            for i in range(2,mop.dim+1):
                yi = x[i-1]-np.power(x[0],0.5*(1.0+3.0*(i-2.0)/(mop.dim-2.0)))
                pi = np.cos(20.0*yi*np.pi/np.sqrt(i+0.0))
               
                if i % 2 == 0:
                    s2 = s2 + yi*yi
                    prod2 = prod2 * pi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi*yi
                    prod1 = prod1 * pi
                    count1 = count1 + 1.0
                    
            f[0] = x[0]+ 2.0*(4.0*s1 - 2.0*prod1 + 2.0) /count1
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0*(4.0*s2 - 2.0*prod2 + 2.0) /count2
            return f

        if case('UF4'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2*x[i-1]
                yi = xi-np.sin(6.0*np.pi*x[0]+i*np.pi/mop.dim)
                hi = np.fabs(yi)/(1.0+np.exp(2.0*np.fabs(yi)))
                
                if i % 2 == 0:
                    s2 = s2 + hi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + hi
                    count1 = count1 + 1.0
                    
            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.power(x[0],2) + 2.0 * s2 /count2
            return f

        if case('UF5'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            N = 10.0
            E = 0.1
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2*x[i-1]
                yi = xi-np.sin(6.0*np.pi*x[0]+i*np.pi/mop.dim)
                hi = 2.0*yi*yi - np.cos(4.0*np.pi*yi) + 1.0
                
                if i % 2 == 0:
                    s2 = s2 + hi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + hi
                    count1 = count1 + 1.0
                    
            hi = (0.5/N + E)*np.fabs(np.sin(2.0*N*np.pi*x[0]));
            f[0] = x[0]	      + hi + 2.0*s1 /count1;
            f[1] = 1.0 - x[0] + hi + 2.0*s2 /count2;
            return f
            
        if case('UF6'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            prod1 = prod2 = 1.0
            N = 2.0
            E = 0.1
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2*x[i-1]
                yi = xi-np.sin(6.0*np.pi*x[0]+i*np.pi/mop.dim)
                pi = np.cos(20.0*yi*np.pi/np.sqrt(i+0.0))
                
                if i % 2 == 0:
                    s2 = s2 + yi*yi
                    prod2 = prod2 * pi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi*yi
                    prod1 = prod1 * pi
                    count1 = count1 + 1.0
                    
            hi = 2.0*(0.5/N + E)*np.sin(2.0*N*np.pi*x[0])
            if hi<0.0:
                hi = 0.0
            f[0] = x[0] + hi + 2.0*(4.0*s1 - 2.0*prod1 + 2.0) /count1
            f[1] = 1.0 - x[0] + hi + 2.0*(4.0*s2 - 2.0*prod2 + 2.0) /count2
            return f            

        if case('UF7'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
                yi = xi - np.sin(6.0*np.pi*x[0] + i*np.pi/mop.dim)
                
                if i % 2 == 0:
                    s2 = s2 + yi*yi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi*yi
                    count1 = count1 + 1.0

            f[0] =  np.power(x[0],0.2)  + 2.0 * s1 / count1 
            f[1] = 1.0 -  np.power(x[0],0.2) + 2.0 * s2 /count2
            return f           

        if case('UF8'):
            s1 = s2 = s3 =  0.0
            count1 = count2 = count3 = 0.0
            
            for i in range(3,mop.dim+1):
                xi = -2.0 + 4.0*x[i-1]
                yi = xi - 2.0*x[1]*np.sin(2.0*np.pi*x[0]+i*np.pi/mop.dim)
                
                if i % 3 == 1:
                    s1 = s1 + yi*yi
                    count1 = count1 + 1.0
                if i % 3 == 2:
                    s2 = s2 + yi*yi
                    count2 = count2 + 1.0
                else:
                    s3 = s3 + yi*yi
                    count3 = count3 + 1.0

            f[0] = np.cos(0.5*np.pi*x[0])*np.cos(0.5*np.pi*x[1]) + 2.0*s1 / count1
            f[1] = np.cos(0.5*np.pi*x[0])*np.sin(0.5*np.pi*x[1]) + 2.0*s2 / count2
            f[2] = np.sin(0.5*np.pi*x[0])                  + 2.0*s3 / count3
            return f   
            
        if case('UF9'):
            s1 = s2 = s3 =  0.0
            count1 = count2 = count3 = 0.0
            E = 0.1
            
            for i in range(3,mop.dim+1):
                xi = -2.0 + 4.0*x[i-1]
                yi = xi - 2.0*x[1]*np.sin(2.0*np.pi*x[0]+i*np.pi/mop.dim)
                
                if i % 3 == 1:
                    s1 = s1 + yi*yi
                    count1 = count1 + 1.0
                if i % 3 == 2:
                    s2 = s2 + yi*yi
                    count2 = count2 + 1.0
                else:
                    s3 = s3 + yi*yi
                    count3 = count3 + 1.0

            yi = (1.0+E)*(1.0-4.0*(2.0*x[0]-1.0)*(2.0*x[0]-1.0))
            if yi<0.0:
               yi = 0.0
            f[0] = 0.5*(yi + 2*x[0])*x[1] + 2.0*s1 /count1
            f[1] = 0.5*(yi - 2*x[0] + 2.0)*x[1] + 2.0*s2 /count2
            f[2] = 1.0 - x[1] + 2.0*s3 /count3
            return f     

        if case('UF10'):
            s1 = s2 = s3 =  0.0
            count1 = count2 = count3 = 0.0
            
            for i in range(3,mop.dim+1):
                xi = -2.0 + 4*x[i-1]
                yi = xi - 2.0*x[1]*np.sin(2.0*np.pi*x[0]+i*np.pi/mop.dim)
                hi = 4.0*yi*yi - np.cos(8.0*np.pi*yi) + 1.0
                
                if i % 3 == 1:
                    s1 = s1 + hi
                    count1 = count1 + 1.0
                if i % 3 == 2:
                    s2 = s2 + hi
                    count2 = count2 + 1.0
                else:
                    s3 = s3 + hi
                    count3 = count3 + 1.0

            f[0] = np.cos(0.5*np.pi*x[0])*np.cos(0.5*np.pi*x[1]) + 2.0*s1 /count1
            f[1] = np.cos(0.5*np.pi*x[0])*np.sin(0.5*np.pi*x[1]) + 2.0*s2 /count2
            f[2] = np.sin(0.5*np.pi*x[0]) + 2.0*s3 /count3
            return f   
         
         
        
        """LZ Test Instances"""
        if case('LZ1'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                theta = 1.0 + 3.0*(i-2)/(mop.dim - 2);
                yi    = x[i-1] - np.power(x[0], 0.5*theta)
                yi    = yi * yi
                
                if i % 2 == 0:
                    s2 = s2 + yi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi
                    count1 = count1 + 1.0

            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f
        
        if case('LZ2'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
                yi = xi - np.sin(6.0*np.pi*x[0] + i*np.pi/mop.dim)
                yi = yi * yi
                
                if i % 2 == 0:
                    s2 = s2 + yi
                    count2 = count2 + 1.0
                else:
                    s1 = s1 + yi
                    count1 = count1 + 1.0

            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f
        
        if case('LZ3'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
                theta = 6.0*np.pi*x[0] + i*np.pi/mop.dim
                
                if i % 2 == 0:
                    yi = xi - 0.8*x[0]*np.cos(theta)
                    s2 = s2 + yi * yi
                    count2 = count2 + 1.0
                else:
                    yi = xi - 0.8*x[0]*np.sin(theta)
                    s1 = s1 + yi * yi
                    count1 = count1 + 1.0
                    
            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f

        if case('LZ4'):
            s1 = s2 =  0.0
            count1 = count2 =  0.0
            
            for i in range(2,mop.dim+1):
                xi = -1.0 + 2.0*x[i-1]
                theta = 6.0*np.pi*x[0] + i*np.pi/mop.dim
                
                if i % 2 == 0:
                    yi = xi - 0.8*x[0]*np.cos(theta/3)
                    s2 = s2 + yi * yi
                    count2 = count2 + 1.0
                else:
                    yi = xi - 0.8*x[0]*np.sin(theta)
                    s1 = s1 + yi * yi
                    count1 = count1 + 1.0
                    
            f[0] = x[0]  + 2.0 * s1 / count1 
            f[1] = 1.0 - np.sqrt(x[0]) + 2.0 * s2 /count2
            return f
        
