#!/usr/bin/env python

import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
import sys
import os
import ast

def readfile(fname):

	X=[]
	Y=[]
	for l in open(fname):
		[x,y]= l[1:].strip(']\n').split(',')
		X.append(map(float,x[2:].strip(']]').split()))
		Y.append([float(y)])

	X=sp.matrix(X)
	Y=sp.matrix(Y)
	return X,Y

def eval_llk(X,Y,kf,sigma):
	tmp=buildKsym(kf,X)
	N=tmp.shape[0]
	tmp+=sigma*sp.eye(N)
	llk= (-Y.T*tmp*Y-0.5*sp.log(spl.det(tmp))-0.5*N*sp.log(2*sp.pi))[0,0]
	return llk

def buildKsym(kf,x):
        #x should be  column vector
        (l,_)=x.shape
        K=sp.matrix(sp.empty([l,l]))
        for i in range(l):
            	K[i,i]=kf(x[i,:].T,x[i,:].T)#+0.0001
            	for j in range(i+1,l):
                	K[i,j]=kf(x[i,:].T,x[j,:].T)
                
                	K[j,i]=K[i,j]
                
        return K

def llk(X,Y,k_gen_fn,para):
	theta=para[:-1]
	sigma=para[-1]
	k=k_gen_fn(theta)
	l=eval_llk(X,Y,k,sigma)
	return l

def gen_sqexp_k(theta):
	A=theta[0]
	d=sp.matrix(theta[1:]).T
	
	N=sp.matrix(d).shape[0]
	D=sp.eye(N)
	for i in range(N):
		D[i,i]=1./(d[i,0]**2)
	return lambda x,y:A*sp.exp(-0.5*(x-y).T*D*(x-y))
	


fname='spares_81/ratio.txt'
X,Y=readfile(fname)


tmp=lambda x:llk(X,Y,gen_sqexp_k,[x,0.001,0.001,0.001,0.001,0.001,0.001])
xax=sp.logspace(-5,2,40)
yax=map(tmp,xax)
plt.plot(xax,yax)
plt.xscale('log')
print llk(X,Y,gen_sqexp_k,[0.001,0.001,0.001,0.001,0.001,0.001,0.001])
plt.show()
