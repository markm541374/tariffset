import scipy as sp
from scipy import interpolate as spi
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time

def buildKsym(kf,x):
       #x should be  column vector
       (l,_)=x.shape
       K=sp.matrix(sp.empty([l,l]))
       for i in range(l):
           K[i,i]=kf(x[i,:],x[i,:])+0.0001
           for j in range(i+1,l):
               K[i,j]=kf(x[i,:],x[j,:])
                
               K[j,i]=K[i,j]
                
       return K
    

def gen(kf,upper,lower,r=25):


	
	X=sp.matrix(sp.ndarray([0,2]))

	xv=sp.linspace(lower[0],upper[0],r)
	yv=sp.linspace(lower[1],upper[1],r)

	for x in xv:
    		for y in yv:
        
        		p=sp.matrix([[x,y]])
        		X=sp.vstack([X,p])

	K= sp.array(buildKsym(kf,X))
	m=sp.zeros([K.shape[0]])

	zv = sp.random.multivariate_normal(m,K)

	f=spi.interp2d(xv,yv,sp.vstack(sp.split(zv,r)))
	g = lambda x:f(x[0,0],x[0,1])[0]
	return g

	
