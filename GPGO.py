import scipy as sp
from scipy.stats import norm as norm
import scipy.linalg as spl
import numpy.linalg as npl #slogdet isn't in spl
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
    def __init__(self,KF_gen,KF_init,KF_prior,optF,upper,lower,dim):
	self.KF_gen=KF_gen
	self.KF_hyp=KF_init
	self.KF_prior=KF_prior
        self.KFs=[KF_gen(self.KF_hyp)]
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
	self.cc=0
	self.finished=False
	self.fudgelimit=32
	self.hyp_trace=sp.matrix(KF_init)
	
        return
    
    def findnext(self):
        if self.nsam==0:
            return [0.5*(sp.matrix(self.upper)+sp.matrix(self.lower)),0]
	if self.finished:
		raise StandardError("opt is finished")
        self.cc=0
	fudge=2.
        EIwrap= lambda x,y : (-self.evalWEI(sp.matrix(x)),0)
        [x,EImin,ierror]=DIRECT.solve(EIwrap,self.lower,self.upper,user_data=[],algmethod=1,maxf=4000)
	while self.cc==0 and fudge<=self.fudgelimit:
		print "non nonzero eis found over full range. trying closer to current min with lengthfactor: "+str(fudge)	
		u=sp.matrix(self.upper)
		l=sp.matrix(self.lower)
		dia=u-l
		lw=sp.maximum(l,self.best[0]-dia/fudge)
		up=sp.minimum(u,self.best[0]+dia/fudge)
		[x,EImin,ierror]=DIRECT.solve(EIwrap,lw,up,user_data=[],algmethod=1,maxf=4000)
		fudge*=2.
		print "nonzero EIs: " +str(self.cc)
	if self.cc==0:
		print "done. no nonzero EIs"
		
		self.finished=True
		#raise StandardError("opt is finished")
		return [self.best[0],0.]
	
        return sp.matrix(x),-EImin
    
   
    def forceAddPoint(self,x,y,suppress_update=False):
        self.nsam+=1
        self.X=sp.vstack([self.X,x])
        self.Y=sp.vstack([self.Y,y])
        delta=sp.matrix([[0.]])
        if y<self.best[1]:
            
            delta=self.best[1]-y
            self.best=[x,y]
	if not suppress_update:
		#print "x"
        	self.Ky=[]
        	self.KyRs=[]
        	self.llks=[]
		
        	for kf in self.KFs:
            		tmp=self.buildKsym(kf,self.X)
            		self.Ky.append(tmp)
            		self.KyRs.append(spl.cho_factor(tmp))
            		self.llks.append(self.eval_kernel_llk(kf))
        	self.renorm=sum(map(sp.exp,self.llks))
		if self.renorm==0.:
			pass#raise ValueError("renorm=0.0")
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
            self.llks.append(self.eval_kernel_llk(kf))
        self.renorm=sum(map(sp.exp,self.llks))
	if self.renorm==0.:
		raise ValueError("renorm=0.0")
        return y

    def eval_kernel_llk(self,kf):
	K=self.buildKsym(kf,self.X)
    	K_lu=spl.cho_factor(K)
    	Ki_Y=spl.cho_solve(K_lu,self.Y)
    	llk= (-self.Y.T*Ki_Y-0.5*npl.slogdet(K)[1]-0.5*self.Y.shape[0]*sp.log(2*sp.pi))[0,0]
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
	#print "alpha: "+str(alpha)
	#print "Z: "+str(Z)
        E=-mu+norm.pdf(alpha)*sigma/Z+ER
        EI=Z*E
        if np.isfinite(EI):
            return sp.matrix(EI)
        else:
            return sp.matrix(0.0)
        
    def evalWEI(self,x):
        n=len(self.KFs)
        j=map(self.evalWEIi,range(n),[x]*n)
        r=sum(j)/self.renorm
        if r>0.:
		self.cc+=1
        
        return r
            
    def evalWEIi(self,i,x):
	
	e=self.evalEI(self.X,self.Y,self.KyRs[i],self.KFs[i],self.best[1],x)[0]
	l=sp.exp(self.llks[i])
	#if e>0.:
	#	print str(l)+"   "+str(e)
        return l*e
        
    def evalEI(self,X,Y,KyR,kf,best,x):
        
        Kss = sp.matrix(kf(x,x))+0.0001
        Ksy = self.buildKasym(kf,X,sp.matrix(x))
        tmp = spl.cho_solve(KyR,Ksy).T
        m=tmp*Y
	
        C=Kss-tmp*Ksy
	
        E=self.EI(best,m[0,0],C[0,0])
	#if E>0.:
	#	print "m: "+str(m)+" c: "+str(C)+" b: "+str(best)
        return (E,m[0,0],C[0,0])
    
    def evalPY(self,x,y):
        p=0.
        for i in range(len(self.KFs)):
            lki = sp.exp(self.llks[i])
            (Ei,mi,Ci) = self.evalEI(self.X,self.Y,self.KyRs[i],self.KFs[i],self.best[1],x)
           
            p+=lki*sp.exp(-((mi-y)**2)/Ci)*(1/(sp.sqrt(Ci*2*sp.pi)))
           
        return p/self.renorm

    def plot_hyper(self):
	r=30
	xaxis=sp.linspace(-2,2,r)
	yaxis=sp.linspace(-2,2,r)
	C=[]
	Cp=[]
	for x in xaxis:
		print x
		for y in yaxis:
			kf=self.KF_gen([1.,sp.exp(x),sp.exp(y)])
			l=self.eval_kernel_llk(kf)
			C.append(10*l)
			Cp.append(sp.exp(l+(-((x/3.)**2+(y/3.)**2))))
	Cgrid=sp.vstack(np.split(np.array(C),r))
	Cpgrid=sp.vstack(np.split(np.array(Cp),r))
	plt.contour(xaxis,yaxis,Cgrid,r)
	plt.figure()
	plt.contour(xaxis,yaxis,Cpgrid,r)
	return
    def eval_kernel_ap(self,kf_para):
	pr=1
	for i,x in enumerate(kf_para):
		[mu,std]=self.KF_prior[i]
		p=norm.pdf(x,loc=mu,scale=std)
		pr*=p
	llk=self.eval_kernel_llk(self.KF_gen(map(lambda x:10**x,kf_para)))
	kf_log_post=llk+sp.log(pr)
	return kf_log_post

    def search_hyper(self):
	Fwrap= lambda  x,y: (-self.eval_kernel_ap(x),0)
	upper=[]
	lower=[]
	for i in self.KF_prior:
		upper.append(i[0]+3*i[1])
		lower.append(i[0]-3*i[1])
	[loghyp,Fmin,ierror]=DIRECT.solve(Fwrap,lower,upper,user_data=[],algmethod=1,maxf=2000)
        return [map(lambda x:10**x,loghyp),-Fmin]

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

    def savehyptrace(self,fname):
        sp.savetxt(fname,self.hyp_trace)
        return

    def loadtrace(self,fname):
        e=sp.loadtxt(fname)
	l=len(e)
        for j,i in enumerate(e):
            self.forceAddPoint(i[0:-1],i[-1],suppress_update=(j!=l-1))
        return

    def search(self,n,init=12):
	t_00=time.time()
	for i in range(n):
		print "----x----x----\n----x----x----\nstep "+str(i+1)+" of "+str(n)+"\n"
		if i==0:
			
			x=0.5*(sp.matrix(self.upper)+sp.matrix(self.lower))
			t1=time.time()
			print "init with center "+str(x)
		elif i<init:
			x=sp.matrix(sp.zeros([1,self.dim]))
			for j in range(self.dim):
				x[0,j]=sp.random.uniform(self.lower[j],self.upper[j])
			print "random eval loaction "+ str(x)
			t1=time.time()
		else:
			t0=time.time()
			print "Searching for best eval location..."
			[x,ei]=self.findnext()
			t1=time.time()
			print "searchtime = "+str(t1-t0)
    			print "Found optimum: " + str(x)
			print "EI at optimum: " + str(-ei)
		print "---"
		print "Evaluating new point"
		print "current best: "+str(self.best[1])
		print "evaluating at EImax..."
		y=self.evaluate(x)
		t2=time.time()
		print "evaluation time = "+str(t2-t1)
		print "evaluated as "+str(y)
		if i==init-1 or (i>init and i<80 and i%10==0) or (i>=80 and i%15==0):
			print "---"
			print "Reopt hyperparameters"
			print "hyperparameters before search: "+str(self.KF_hyp)
			print "lap before search: "+str(self.eval_kernel_ap(map(sp.log10,self.KF_hyp)))
			print "searching for map hyperparameters..."
			[h,l]=self.search_hyper()
			t3=time.time()
			print "searchtime = "+str(t3-t2)
		
			print "map hyperparameters: "+str(h)
			print "lmap : "+str(l)
			self.KF_hyp=h
			self.hyp_trace=sp.vstack([self.hyp_trace,h])
		print "running time so far: "+str(time.time()-t_00)
		print "----x----x----\n----x----x----\n"
		self.savetrace("defaulttrace.txt~")
		self.savehyptrace("defaulthyptrace.txt~")
		plt.show()
		
		if self.finished:
			break
	print "\nxxxx-xxxx-xxxx\n"
	if self.finished:
		print "terminated after "+str(i+1)+" evaluations"
	else:
		print "scheduled end after "+str(i+1)+" evaluations"
	print "min: "+str(self.best[1])
	print "argmin: "+str(self.best[0])
	return self.best
		
	
    def stepn(self,n,h_opt=False):
	t_00=time.time()
	for i in range(n):
		print "----x----x----\n----x----x----\nstep "+str(i+1)+" of "+str(n)+"\n"
		t0=time.time()
		print "Searching for best eval location..."
		[x,ei]=self.findnext()
		t1=time.time()
		print "searchtime = "+str(t1-t0)
    		print "Found optimum: " + str(x)
		print "EI at optimum: " + str(-ei)
		print "---"
		print "Evaluating new point"
		print "current best: "+str(self.best[1])
		print "evaluating at EImax..."
		y=self.evaluate(x)
		t2=time.time()
		print "evaluation time = "+str(t2-t1)
		print "evaluated as "+str(y)
		if h_opt:
			print "---"
			print "Reopt hyperparameters"
			print "hyperparameters before search: "+str(self.KF_hyp)
			print "lap before search: "+str(self.eval_kernel_ap(map(sp.log10,self.KF_hyp)))
			print "searching for map hyperparameters..."
			[h,l]=self.search_hyper()
			t3=time.time()
			print "searchtime = "+str(t3-t2)
		
			print "map hyperparameters: "+str(h)
			print "lmap : "+str(l)
			self.KF_hyp=h
			self.hyp_trace=sp.vstack([self.hyp_trace,h])
		print "running time so far: "+str(time.time()-t_00)
		print "----x----x----\n----x----x----\n"
		self.savetrace("defaulttrace.txt~")
		self.savehyptrace("defaulthyptrace.txt~")
		plt.show()
		
		if self.finished:
			break
	print "\nxxxx-xxxx-xxxx\n"
	if self.finished:
		print "terminated after "+str(i+1)+" evaluations"
	else:
		print "scheduled end after "+str(i+1)+" evaluations"
	print "min: "+str(self.best[1])
	print "argmin: "+str(self.best[0])
	return self.best
