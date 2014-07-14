#!/usr/bin/env python

import scipy as sp
from scipy import random as spr
import numpy as sp
from matplotlib import pyplot as plt
import operator
import ast

import sys
from multiprocessing import Pool, Process, Pipe
import os

os.system("taskset -p 0xff %d" % os.getpid())

#all times are in seconds


class agent:
	def __init__(self):
		#init the agent with its goals
		return
	def inform_tariff(self,T):
		#informs the agent of a new tariff. T should be a function accepting time(s) and returning cost (pounds/W)
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
		self.lv=para[3] #value of the load when running
		self.uf=para[4] #utility gradient for deferal
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
			cost_acc+=self.lv*self.T(self.t0+dt+i*60)
		u = -cost_acc-dt*self.uf
		
		return u

	def schedule(self):
		
		#only optimise to the nearest minute
		assert(self.tf>self.t0+self.ld)
		best_c = -sp.Inf
		best_dt = -2
		
		tsl = [i*60 for i in range(0,(self.tf-self.t0-self.ld)/60)]
		
		cs=[0]*len(tsl)
		
		cs=map(self.evaluate_u,tsl)
		for c in cs:
			
			if c>best_c:
				best_c=c
				best_dt=i*60
		
		self.ts=self.t0+best_dt
		return self.ts

	def plotm(self):
		T=[i/60. for i in range(0,24*60)]
		u = []
		tar = []
		for t in [60*60*i for i in T]:
			tar.append(self.T(t)*3600) #multiply by 3600 so that 1W for 1 hour shows as 1
			u.append(self.evaluate_u(t-self.t0))
		plt.figure()
		plt.plot(T,u)
		plt.plot(T,tar)
		plt.show()
		return

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
			self.A.append(rect_SSL_agent(ast.literal_eval(line.strip('\n'))))
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
		for i,a in enumerate(self.A):
			ts = Procs[i][1].recv()
			self.A[i].ts=ts
			Procs[i][0].join()
		return 

	def get_load_m(self):	
		load = [0.]*(24*60)
		for a in self.A:
			load=map(operator.add,load,a.get_load_m())
		return load

def create_SSL_agents(fname):
	f = open(fname,"w")
	N=25
	for i in range(N):
		t0=int(min(3600*spr.gamma(6,1),23*3600)) #time the agent can start
		tf=int(min(t0+3600*(1+spr.gamma(6,1)),24*3600)) #time the agent must finish by (s)
		ld=3600 #time the agent runs for (s)
		lv= 1.#value of the load when running
		uf=spr.gamma(1,1)*0.5/3600. #utility gradient for deferal
		a=rect_SSL_agent([t0,tf,ld,lv,uf])
		f.write(str(a.get_agentpara())+'\n')
	f.close()
	return

def load_cost_flatness(load):
	#squared normalised difference from a flat load
	mean=sum(load)/float(len(load))
	err = sum([((i-mean)/mean)**2 for i in load])/len(load)
	return err

def gen_SG_tariff(theta):
		#sum of gaussians spaced at support points
		N=len(theta)
		spoints = [i*24*3600/float(N-1) for i in range(N)]
		tariff = lambda t:sum([theta[i]*sp.exp(-((t-spoints[i])**2)/float(2*(24*3600/float(N))**2)) for i in range(N)])
		return tariff


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
	
	

	def eval_under_tariff(self,tariffpara,plot_=False):
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
			tm=range(24*60)
			plt.figure()
			plt.plot(tm,map(tariff,[i*60 for i in tm]))
		
			plt.figure()
			for i in range(self.N):
				plt.plot(tm,loads[i])
			
			plt.show()
		
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
	#create_SSL_agents("ts0.txt")
	#create_SSL_agents("ts1.txt")
	#create_SSL_agents("ts2.txt")
	
	o = objective(["ts0.txt","ts1.txt","ts2.txt"],load_cost_flatness,gen_SG_tariff)
	trf=[1,4,1,1,2,1,1.5,1]
	print o.eval_under_tariff(trf,plot_=True)
	
	#load = o.AS.get_load_m()
	#tm=range(24*60)
	#plt.figure()
	#plt.plot(tm,map(gen_SG_tariff(trf),[i*60 for i in tm]))
	#plt.figure()
	#plt.plot(tm,load)
	
	#plt.show()
	
	return

main()


