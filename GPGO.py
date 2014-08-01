import scipy as sp
from scipy.stats import norm as norm
import scipy.linalg as spl
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import DIRECT
from functools import partial
import time
from multiprocessing import Pool

os.system("taskset -p 0xff %d" % os.getpid())
import copy

class GPGO():
    def __init__(self,KFs,optF,upper,lower,dim):
        self.KFs=KFs
        self.optF=optF
        self.upper=upper
        self.lower=lower
        self.nsam=0
        self.X=sp.matrix(sp.ndarray([0,dim]))
        self.Y=sp.matrix(sp.ndarray([0,1]))
        self.best=[None,sp.Inf]
        self.dim=dim
        self.Ky=[]
        self.KyRs=[]
        self.llks=[]
        self.renorm=1.
        return
    
    def findnext(self):
        if self.nsam==0:
            return 0.5*(sp.matrix(self.upper)+sp.matrix(self.lower))
        
        EIwrap= lambda x,y : (-self.evalWEI(sp.matrix(x)),0)
        [x,EImin,ierror]=DIRECT.solve(EIwrap,self.lower,self.upper,user_data=[],algmethod=1,maxf=1500)
        return [sp.matrix(x),-EImin]
    
   
    def forceAddPoint(self,x,y):
        self.nsam+=1
        self.X=sp.vstack([self.X,x])
        self.Y=sp.vstack([self.Y,y])
        delta=sp.matrix([[0.]])
        if y<self.best[1]:
            
            delta=self.best[1]-y
            self.best=[x,y]
        self.Ky=[]
        self.KyRs=[]
        self.llks=[]
        for kf in self.KFs:
            tmp=self.buildKsym(kf,self.X)
            self.Ky.append(tmp)
            self.KyRs.append(spl.cho_factor(tmp))
            self.llks.append((-self.Y.T*tmp*self.Y-0.5*sp.log(spl.det(tmp))-0.5*self.nsam*sp.log(2*sp.pi))[0,0])
        self.renorm=sum(map(sp.exp,self.llks))
	if self.renorm==0.:
		raise ValueError("renorm=0.0")
        return delta
    
    def evaluate(self,x):
        y=self.optF(x)
        self.nsam+=1
        self.X=sp.vstack([self.X,x])
        self.Y=sp.vstack([self.Y,y])
        if y<self.best[1]:
            self.best=[x,y]
        self.Ky=[]
        self.KyRs=[]
        self.llks=[]
        for kf in self.KFs:
            tmp=self.buildKsym(kf,self.X)
            self.Ky.append(tmp)
            self.KyRs.append(spl.cho_factor(tmp))
            self.llks.append((-self.Y.T*tmp*self.Y-0.5*sp.log(spl.det(tmp))-0.5*self.nsam*sp.log(2*sp.pi))[0,0])
        self.renorm=sum(map(sp.exp,self.llks))
	if self.renorm==0.:
		raise ValueError("renorm=0.0")
        return y
    def eval_kernel_llk(self,kf):
	tmp=self.buildKsym(kf,self.X)
	llk= (-self.Y.T*tmp*self.Y-0.5*sp.log(spl.det(tmp))-0.5*self.nsam*sp.log(2*sp.pi))[0,0]
	return llk

    def plotstate1D(self,llimit,ulimit,allK=False):
        assert self.dim==1
        size=100
        Esum=sp.zeros([size,])
        Msum=sp.zeros([size,])
        renorm=0
        for i,kf in enumerate(self.KFs):
            KyR = spl.cho_factor(self.buildKsym(kf,self.X))
            factor = sp.exp(self.llks[i])
            renorm+=factor
            xaxis=sp.linspace(llimit[0],ulimit[0],size)
            
            E=[]
            m=[]
            C=[]
            
            for x in xaxis:
                (Et,mt,Ct)=self.evalEI(self.X,self.Y,KyR,kf,self.best[1],sp.matrix([x]))
                E.append(Et[0,0])
                m.append(mt)
                C.append(Ct)
            
            
           
            Esum=Esum+np.array(E)*factor
            Msum=Msum+np.array(m)*factor
            
        Msum=Msum/renorm
        
        plt.figure(figsize=(14,5))
        plt.subplot(121)
        try:
            
            plt.plot(xaxis,Msum)
        except ValueError:
            pass
        plt.subplot(122)
        
        plt.plot(xaxis,Esum)
        plt.show()
        return
    
    def plotstate2D(self,llimit,ulimit,allK=False):
        assert self.dim==2
        size=60
        Esum=sp.zeros([size,size])
        Msum=sp.zeros([size,size])
        for i,kf in enumerate(self.KFs):
            KyR = spl.cho_factor(self.buildKsym(kf,self.X))
            
            xaxis=sp.linspace(llimit[0],ulimit[0],size)
            yaxis=sp.linspace(llimit[1],ulimit[1],size)
            E=[]
            m=[]
            C=[]
            for y in yaxis:
                for x in xaxis:
                    (Et,mt,Ct)=self.evalEI(self.X,self.Y,KyR,kf,self.best[1],sp.matrix([x,y]))
                    E.append(Et[0,0])
                    m.append(mt)
                    C.append(Ct)
            
            Egrid=sp.vstack(np.split(np.array(E),size))
            mgrid=sp.vstack(np.split(np.array(m),size))
            Cgrid=sp.vstack(np.split(np.array(C),size))
            
            Esum=Esum+Egrid*sp.exp(self.llks[i])
            Msum=Msum+mgrid*sp.exp(self.llks[i])
            if allK:
                plt.figure(figsize=(20,5))
                plt.subplot(131)
                try:
                    plt.contour(xaxis,yaxis,mgrid,50)
                except ValueError:
                    pass
                plt.subplot(132)
                plt.contour(xaxis,yaxis,Cgrid,50)
                plt.subplot(133)
                plt.contour(xaxis,yaxis,Egrid,50)
        
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        try:
            plt.contour(xaxis,yaxis,Msum,50)
        except ValueError:
            pass
        plt.subplot(133)
        plt.contour(xaxis,yaxis,Esum,50)
        plt.show()
        return
    
    def buildKasym(self,kf,x1,x2):
        #x should be column vectors, returns a matrix with x1 rows, x2 columns
        (l1,_)=x1.shape
        (l2,_)=x2.shape
        K=sp.matrix(sp.empty([l1,l2]))
        for i in range(l1):
            for j in range(l2):
                K[i,j]=kf(x1[i,:],x2[j,:])
                
        return K

    def buildKsym(self,kf,x):
        #x should be  column vector
        (l,_)=x.shape
        K=sp.matrix(sp.empty([l,l]))
        for i in range(l):
            K[i,i]=kf(x[i,:],x[i,:])+0.0001
            for j in range(i+1,l):
                K[i,j]=kf(x[i,:],x[j,:])
                
                K[j,i]=K[i,j]
                
        return K
    
    def EI(self,ER,mu,sigma):
        alpha=(-ER+mu)/sigma
        Z=norm.cdf(-alpha)
        if Z==0.0:
            return sp.matrix(0.0)
        E=-mu+norm.pdf(alpha)*sigma/Z+ER
        EI=Z*E
        if np.isfinite(EI):
            return sp.matrix(EI)
        else:
            return sp.matrix(0.0)
        
    def evalWEI(self,x):
        n=len(self.KFs)
        j=map(self.evalWEIi,range(n),[x]*n)
        
            
        
        return sum(j)/self.renorm
            
    def evalWEIi(self,i,x):
        return sp.exp(self.llks[i])*self.evalEI(self.X,self.Y,self.KyRs[i],self.KFs[i],self.best[1],x)[0]
        
    def evalEI(self,X,Y,KyR,kf,best,x):
        
        Kss = sp.matrix(kf(x,x))+0.0001
        Ksy = self.buildKasym(kf,X,sp.matrix(x))
        tmp = spl.cho_solve(KyR,Ksy).T
        m=tmp*Y
        C=Kss-tmp*Ksy
        E=self.EI(best,m[0,0],C[0,0])
        return (E,m[0,0],C[0,0])
    
    def evalPY(self,x,y):
        p=0.
        for i in range(len(self.KFs)):
            lki = sp.exp(self.llks[i])
            (Ei,mi,Ci) = self.evalEI(self.X,self.Y,self.KyRs[i],self.KFs[i],self.best[1],x)
           
            p+=lki*sp.exp(-((mi-y)**2)/Ci)*(1/(sp.sqrt(Ci*2*sp.pi)))
           
        return p/self.renorm
    
    def predictY(self,x):
        res=[]
        for i in range(len(self.KFs)):
            lki = sp.exp(self.llks[i])
            (Ei,mi,Ci) = self.evalEI(self.X,self.Y,self.KyRs[i],self.KFs[i],self.best[1],x)
            res.append([mi,Ci,lki])
        return res
    
    def savetrace(self,fname):
        sp.savetxt(fname,sp.hstack([self.X,self.Y]))
        return
    
    def loadtrace(self,fname):
        e=sp.loadtxt(fname)
        for i in e:
            self.forceAddPoint(i[0:-1],i[-1])
        return
