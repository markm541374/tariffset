#!/usr/bin/env python

import scipy as sp
from scipy import random as spr
from scipy import linalg as spl
from scipy.interpolate import interp1d as spi1
import numpy as sp
from matplotlib import pyplot as plt
import operator
import ast
from functools import partial
import sys
from multiprocessing import Pool, Process, Pipe
import os
import GPGO
import time
from cvxopt import matrix as cm
from cvxopt import solvers as cs

os.system("taskset -p 0xff %d" % os.getpid())
os.environ['OPENBLAS_NUM_THREADS']='1'

#all times are in seconds


class agent:
	def __init__(self):
		#init the agent with its goals
		return
	def inform_tariff(self,T):
		#informs the agent of a new tariff. T should be a function accepting time(s) and returning cost (pounds/unithour)
		self.T=T
		return
	
	def get_total(self):
		#find the total power/cost the agent will use
		return
	def get_agentpara(self):
		return #parameter vector that fully defines the agent



#agent with a single fixed load that runs at a constant level for a fixed time
class rect_SSL_agent(agent):
	def __init__(self,para):
		self.t0=para[0] #time the agent can start
		self.tf=para[1] #time the agent must finish by (s)
		self.ld=para[2] #time the agent runs for (s)
		self.lv=para[3] #value of the load when running (W)
		self.uf=para[4] #utility gradient for deferal pounds_eq/sec
		assert(self.tf<=24*60*60)
		return

	def get_agentpara(self):
		para = [self.t0,self.tf,self.ld,self.lv,self.uf]
		return para
	
	
	def get_load_s(self,t0=0,t1=24*3600):
		#assume a 24 hour period from 0
		lp = [0.]*(24*3600)
		for i in range(self.ts,self.ts+self.ld):
			lp[i]=self.lv
		return lp

	def get_load_m(self,t0=0,t1=24*60):
		#assume a 24 hour period from 0. less accurate as times are rounded to the minute
		lp = [0.]*(24*60)
		for i in range(self.ts/60,(self.ts+self.ld)/60):
			lp[i]=self.lv
		return lp

	def evaluate_u(self,dt):
		
		#find the utility of a given deferal ####per munite accuracy
		if self.t0+dt+self.ld>=self.tf:
			return -sp.Inf
		if dt<0:
			return -sp.Inf
		cost_acc=0
		for i in range(self.ld/60):
			cost_acc+=self.lv*self.T(self.t0+dt+i*60.)/60.
		u = -cost_acc-dt*self.uf
		#print "dt: "+str(dt)+" u: "+str(u)
		return u

	def schedule(self):
		
		#only optimise to the nearest minute
		assert(self.tf>self.t0+self.ld)
		best_c = -sp.Inf
		best_dt = -2
		
		tsl = [i*60 for i in range(0,(self.tf-self.t0-self.ld)/60)]
		
		cs=[0]*len(tsl)
		
		cs=map(self.evaluate_u,tsl)
		for i,c in enumerate(cs):
			
			if c>best_c:
				best_c=c
				best_dt=i*60
		#print "best_c: "+str(best_c)+" best_dt: "+str(best_dt)
		self.ts=self.t0+best_dt
		
		return [self.ts]

	def set_schedule_result(self,para):
		self.ts=para[0]
		return

	def plotm(self):
		N=24*60
		Lambda = sp.matrix(map(self.T,[i*60 for i in range(N)])).T
		Load = sp.matrix(sp.zeros([N,1]))
		Fs = sp.matrix(sp.zeros([N,1]))
		for i in range(self.ts/60,(self.ts+self.ld)/60):
			Load[i,0]=self.lv
		for i in range(self.ts/60,self.tf/60):
			Fs[i,0]=1

		plt.figure(figsize=(18,12))
		ax1 = plt.subplot2grid((3,5),(0,0),rowspan=1,colspan=5)
		ax1.set_ylabel("Load (W)")
		ax1.plot(sp.array(Load))
		ax1.axis([0,N,0,1])

		ax2 = plt.subplot2grid((3,5),(1,0),rowspan=1,colspan=5)
		ax2.set_ylabel("Feasible")
		ax2.plot(sp.array(Fs))
		ax2.axis([0,N,0,2])
		plt.draw()

		ax3 = plt.subplot2grid((3,5),(2,0),rowspan=1,colspan=5)
		ax3.set_ylabel("Tariff")
		ax3.plot(sp.array(Lambda))
		ax3.axis([0,N,0,40])
		plt.draw()
		return

class quadthermo_agent(agent):
	def __init__(self,para):
		self.absmax=para[0]
		self.absmin=para[1]
		#activetimes is a list of blocks in which the heating is on. each element is [tstart,tstop,Ttarget,margin]
		self.activetimes=para[2]
		self.q=para[3]
		#Text is a function accepting time (seconds) and returning temp
		self.Textfn=lambda x:1
		self.P=para[4]
		self.cm=para[5]
		self.k=para[6]
		self.dt=60
		self.tmpl=[self.k]
		return

	def get_agentpara(self):
		para = [self.absmax,self.absmin,self.activetimes,self.q,self.P,self.cm,self.k]
		return para

	def schedule(self):
		#number of thermal sim periods
		N=24*60*60/self.dt
		#number of pwn periods
		M=4*24
		#pwm/thermal period ratio
		J=N/M
		#Te is the vector of external temperatures
		Te=sp.matrix(map(self.Textfn,range(N))).T
		#Lambda is the vector of tariffs
		Lambda = sp.matrix(map(self.T,[i*60 for i in range(N)])).T
		#Q is the zero matrix with the cost of errors on the leading diagonal
		Q=sp.matrix(sp.zeros([N,N]))
		#Ts is the target temperature
		Ts=sp.matrix(sp.zeros([N,1]))

		for onperiod in self.activetimes:
			start=onperiod[0]
			stop=onperiod[1]
			Tt=onperiod[2]
			for i in range(start,stop):
				Ts[i,0]=Tt
				Q[i,i]=self.q
		
		U=sp.matrix(sp.vstack([sp.hstack([sp.zeros([1,N-1]),sp.ones([1,1])]),sp.hstack([sp.eye(N-1),sp.zeros([N-1,1])])]))
		
		PhiI = float(self.cm)/(self.dt*self.P) * (sp.eye(N)-(1-self.dt*self.k/self.cm)*U)
		PhiLU = spl.lu_factor(PhiI)
		Psi=self.cm*spl.lu_solve(PhiLU, (self.dt*self.k/self.cm*U*Te))/(self.P*self.dt)
		D = sp.kron(sp.eye(M),sp.ones([J,1]))
		PhiD = spl.lu_solve(PhiLU,D)
		R = PhiD.T*Q*PhiD
		St = 2*(Psi-Ts).T*Q*PhiD + self.P*Lambda.T*D
		
		K=sp.matrix(sp.vstack([sp.eye(M),-sp.eye(M),-PhiD,PhiD]))
		Y=sp.matrix(sp.vstack([sp.ones([M,1]),sp.zeros([M,1]),-self.absmin*sp.ones([N,1])+Psi,self.absmax*sp.ones([N,1])-Psi]))
		
		for onperiod in self.activetimes:
			start=onperiod[0]
			stop=onperiod[1]
			Tt=onperiod[2]
			Margin=onperiod[3]
			for i in range(start,stop):
				e=sp.matrix(sp.zeros([N,1]))
				e[i,0]=1
				K=sp.vstack([K,-e.T*PhiD,e.T*PhiD])
				Y=sp.vstack([Y,-sp.matrix(Tt-Margin)+e.T*Psi,sp.matrix(Tt+Margin)-e.T*Psi])
		
		Qd = 2*cm(R)
		p = cm(sp.array(St).ravel())
		G = cm(K)
		h = cm(sp.array(Y).ravel())
		
		f0=sys.stdout
		f1=open(os.devnull,'w')
		sys.stdout=f1

		sol=cs.qp(Qd, p, G, h)
		u=sp.matrix(sol['x'])

		sys.stdout=f0
		 
		Ti=PhiD*u+Psi
		#print self.dt*(Ti-Ts).T*Q*(Ti-Ts)/self.q
		#print self.dt*self.P*Lambda.T*D*u
		self.Ti=Ti
		self.Te=Te
		self.Ts=Ts
		delta=D*u
		self.delta=delta
		qv=sp.diag(Q)
		self.qv=qv
		self.N=N
		self.Lambda=Lambda
		#self.tmpl.append((60*self.P*self.Lambda.T*delta)[0,0])
		#print self.tmpl

		####
		#self.plotm()
		
		####
		return [Ti,Te,Ts,delta,qv,N,Lambda]

	def set_schedule_result(self,para):
		self.Ti=para[0]
		self.Te=para[1]
		self.Ts=para[2]
		self.delta=para[3]
		self.qv=para[4]
		self.N=para[5]
		self.Lambda=para[6]
		
		return

	def plotm(self):
		plt.figure(figsize=(18,12))

		ax0 = plt.subplot2grid((8,5),(0,0),rowspan=4,colspan=5)
		ax0.set_ylabel("Temperature (C)")
		ax0.plot(sp.array(self.Ti))
		ax0.plot(sp.array(self.Ts))
		ax0.plot(sp.array(self.Te))
		ax0.axis([0,self.N,0,22])

		ax1 = plt.subplot2grid((8,5),(4,0),rowspan=1,colspan=5)
		ax1.set_ylabel("Load (kW)")
		ax1.plot(sp.array(0.001*self.P*self.delta))
		#ax1.plot(sp.array(self.Lambda))
		ax1.axis([0,self.N,0,0.001*self.P*0.5])

		ax2 = plt.subplot2grid((8,5),(5,0),rowspan=1,colspan=5)
		ax2.set_ylabel("Tariff (pence per kWh)")
		ax2.plot(sp.array(60*60*1000*self.Lambda))
		ax2.axis([0,self.N,0,0.5])
		ax2.set_xlabel("time (min)")
		plt.savefig("tmp.eps")
		plt.draw()
		return

	def get_load_m(self):
		#P is watts so *60 for energy/minute
		return self.P*60.*sp.ravel(self.delta.flatten())

def multischedule(conn,agent):
	ts = agent.schedule()
	conn.send(ts)
	conn.close()
	return ts
	
class agent_set():
	def __init__(self):
		self.A=[]
		return
	def write_agents(self,fname):
		f = open(fname,"w")
		for a in self.A:
			f.write(str(a.get_agentpara())+'\n')
		f.close()

	def read_agents(self,fname):
		self.A=[]
		f=open(fname)
		for line in f:
			a=ast.literal_eval(line.strip('\n'))
			if a[0]=="rect_SSL_agent":
				self.A.append(rect_SSL_agent(a[1:]))
			if a[0]=="quadthermo_agent":
				self.A.append(quadthermo_agent(a[1:]))
		return

	def set_tariff(self,tariff):
		for a in self.A:
			a.inform_tariff(tariff)
		return

	def schedule(self):
		Procs=[0]*len(self.A)
		for i,a in enumerate(self.A):
			pc,cc=Pipe()
			p = Process(target=multischedule, args=(cc,a,))
			p.start()
			Procs[i]=[p,pc]
		time.sleep(4)
		for i,a in enumerate(self.A):
			
			ans = Procs[i][1].recv()
			self.A[i].set_schedule_result(ans)
			
			Procs[i][0].join()
			
			#self.A[i].plotm()
		
		return 

	def get_load_m(self):	
		load = [0.]*(24*60)
		for a in self.A:
			load=map(operator.add,load,a.get_load_m())
		return load



def load_cost_sqflatness(load):
	#squared normalised difference from a flat load
	mean=sum(load)/float(len(load))
	err = sum([((i-mean)/mean)**2 for i in load])/len(load)
	return err

def load_cost_absmax(load):
	return max(load)

def load_cost_abflatness(load):
	mean=sum(load)/float(len(load))
	err = sum([abs((i-mean)/mean) for i in load])/len(load)
	return err

def load_cost_ratio(load):
	return max(load)/float(min(load))

def gen_SG_tariff(theta):
		#sum of gaussians spaced at support points
		#theta is pounds/khW output is pounds/Joule
		N=len(theta)
		if N==1:
			theta=sp.array(theta).ravel()
			N=len(theta)
		spoints = [i*24*3600/float(N-1) for i in range(N)]
		tariff = lambda t:(1./(60.*60.*1000.))*sum([theta[i]*sp.exp(-((t-spoints[i])**2)/float(2*(24*3600/float(N))**2)) for i in range(N)])
		return tariff

def gen_3interp_meancon_tariff(theta):
	
	N=len(theta)
	
	
	if N==1:
		theta=sp.array(theta).ravel()
		N=len(theta)
	s=sum(theta)
	av=s/float(N)
	off=0.12-av
	theta=sp.array(map(lambda x:x+off, theta))
	
	spoints = [i*24*3600/float(N-1) for i in range(N)]
	f=spi1(spoints,(1./(60.*60.*1000.))*theta,kind='cubic')
	return f

def gen_3interp_tariff(theta):
	N=len(theta)
	if N==1:
		theta=sp.array(theta).ravel()
		N=len(theta)
	spoints = [i*24*3600/float(N-1) for i in range(N)]
	f=spi1(spoints,(1./(60.*60.*1000.))*sp.array(theta),kind='cubic')	
	a0=0.4*(1./(60.*60.*1000.))
	a1=0.05*(1./(60.*60.*1000.))
	g=lambda x:min(a0,max(a1,f(x)))
	return g

class objective():
	def __init__(self,afnames,loadcostfn,tariffgen):
		self.N=len(afnames)
		self.ASs=[]
		for i in range(self.N):
			self.ASs.append(agent_set())
			self.ASs[i].read_agents(afnames[i])
		self.loadcostfn=loadcostfn
		self.tariffgen=tariffgen
		return
	
	

	def eval_under_tariff(self,tariffpara,plot_=False,data_=False):
		tariff=self.tariffgen(tariffpara)
		cost=0.
		loads=[]
		
		for i in range(self.N):
			self.ASs[i].set_tariff(tariff)
		
		
		map(agent_set.schedule,self.ASs)
		

		for i in range(self.N):
			load=self.ASs[i].get_load_m()
			loads.append(load)
			cost+=self.loadcostfn(load)
		cost=cost/float(self.N)
		if plot_:
			data=[]
			
			tm=range(24*60)
			plt.figure(figsize=(18,12))
			ax0 = plt.subplot2grid((8,5),(3,0),rowspan=1,colspan=5)
			ax1 = plt.subplot2grid((8,5),(0,0),rowspan=3,colspan=5)

			ax1.set_ylabel("Load")
			ax0.set_ylabel("Tariff")
			tr=map(lambda x:360000000*tariff(x),[j*60 for j in tm])
			ax0.plot(tm,tr)
			ax0.axis([0,1600,0,50])
			data.append(tr)
			for i in range(self.N):
				ax1.plot(tm,loads[i])
				data.append(loads[i])
				
			plt.draw()
		sqflatness_cost=0
		abflatness_cost=0
		absmax_cost=0
		ratio_cost=0
		for i in range(self.N):
			load=self.ASs[i].get_load_m()
			loads.append(load)
			sqflatness_cost+=load_cost_sqflatness(load)
			abflatness_cost+=load_cost_abflatness(load)
			absmax_cost+=load_cost_absmax(load)
			ratio_cost+=load_cost_ratio(load)
		sqflatness_cost=sqflatness_cost/float(self.N)
		abflatness_cost=abflatness_cost/float(self.N)
		absmax_cost=absmax_cost/float(self.N)
		ratio_cost=ratio_cost/float(self.N)
		f0=open('spares/sqflatness.txt','a+')
		f1=open('spares/abflatness.txt','a+')
		f2=open('spares/absmax.txt','a+')
		f3=open('spares/ratio.txt','a+')
		f0.write('['+str(tariffpara)+','+str(sqflatness_cost)+']\n')	
		f1.write('['+str(tariffpara)+','+str(abflatness_cost)+']\n')
		f2.write('['+str(tariffpara)+','+str(absmax_cost)+']\n')
		f3.write('['+str(tariffpara)+','+str(ratio_cost)+']\n')
		f0.close()
		f1.close()
		f2.close()
		f3.close()
		if data_:
			return cost,data
		return cost

	def flat_ref(self):
		tariff = lambda x:1.
		cost=0.
		loads=[]
		for i in range(self.N):
			self.ASs[i].set_tariff(tariff)
			self.ASs[i].schedule()
			load=self.ASs[i].get_load_m()
			loads.append(load)
			cost+=self.loadcostfn(load)

		tm=range(24*60)
		plt.figure()
		plt.plot(tm,map(tariff,[i*60 for i in tm]))
		plt.figure()
		for i in range(self.N):
			plt.plot(tm,loads[i])
		plt.show()
		return cost


def main():
	
	o = objective(["ts0.txt"],load_cost_abflatness,gen_SG_tariff)
	trf=[0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10]
	o.eval_under_tariff(trf,plot_=False)
	o.ASs[0].A[0].plotm()
	
	plt.show()
	#print "x"
	#load = o.AS.get_load_m()
	#tm=range(24*60)
	#plt.figure()
	#plt.plot(tm,map(gen_SG_tariff(trf),[i*60 for i in tm]))
	#plt.figure()
	#plt.plot(tm,load)
	
	#plt.show()
	
	return

class testopt():
	def __init__(self,kf_gen,kf_init,kf_prior,o,upper,lower):
		
		self.objective=o
		self.upper=upper
		self.lower=lower
		self.dim=len(upper)
		self.G=GPGO.GPGO(kf_gen,kf_init,kf_prior,self.objective,self.upper,self.lower,self.dim)
		return

	def loadtrace(self,fname):
		self.G.loadtrace(fname)
		return

	def savetrace(self,fname):
		self.G.savetrace(fname)
		return

	def stepn(self,n):
		best=self.G.stepn(n)
		return best
	def search(self,n):
		best=self.G.search(n)
		return best
class experiment():
	def __init__(self,kf_gen,kf_init,kf_prior,ensemblefnames,upper,lower,tariffgen,load_u_fn):
		
		self.ensemblefnames=ensemblefnames
		self.upper=upper
		self.lower=lower
		self.dim=len(upper)
		self.o=objective(ensemblefnames,load_u_fn,tariffgen)
		self.objective=lambda X:self.o.eval_under_tariff(X,plot_=False)
		
		self.G=GPGO.GPGO(kf_gen,kf_init,kf_prior,self.objective,self.upper,self.lower,self.dim)
		return

	def loadtrace(self,fname):
		self.G.loadtrace(fname)
		return

	def savetrace(self,fname):
		self.G.savetrace(fname)
		return
	def savehyptrace(self,fname):
		self.G.savehyptrace(fname)
		return
	def stepn(self,n):
		self.G.stepn(n)
		return
	def search(self,n):
		self.G.search(n)
		return
	def plotflatresponse(self):
		y=self.o.flat_ref()
		print "flatresponse: "+str(y)
		return

	def plotbestresponse(self):
		y=self.objective(self.G.best[0])
		print "bestresponse: "+str(y)
		return
if __name__=="__main__":
	main()
