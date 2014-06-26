#!/usr/bin/env python

import scipy as sp
import numpy as sp
from matplotlib import pyplot as plt
import operator
import ast

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
		print "a_s"
		#only optimise to the nearest minute
		assert(self.tf>self.t0+self.ld)
		best_c = -sp.Inf
		best_dt = -2
		
		for i in range(0,(self.tf-self.t0-self.ld)/60):
			
			c = self.evaluate_u(i*60)
			if c>best_c:
				best_c=c
				best_dt=i*60
		
		self.ts=self.t0+best_dt
		return

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

	def schedule_agents(self):
		for a in self.A:
			a.schedule()
		return

	def set_SG_tariff(self,theta):
		#sum of gaussians spaced at support points
		N=len(theta)
		
		spoints = [i*24*3600/float(N-1) for i in range(N)]
		tariff = lambda t:sum([theta[i]*sp.exp(-((t-spoints[i])**2)/float(2*(24*3600/float(N))**2)) for i in range(N)])
		for a in self.A:
			a.inform_tariff(tariff)
		return tariff

	def schedule(self):
		for a in self.A:
			a.schedule()

	def get_load_m(self):	
		load = [0.]*(24*60)
		for a in self.A:
			load=map(operator.add,load,a.get_load_m())
		return load
	
def main():
	AS = agent_set()
	AS.read_agents('Adef.txt')
	tr=AS.set_SG_tariff([0.25,0.15,0.13,0.12,0.12,0.11,0.1,0.1])
	AS.schedule()
	load = AS.get_load_m()
	tm=range(24*60)
	plt.figure()
	plt.plot(tm,map(tr,[i*60 for i in tm]))
	plt.figure()
	plt.plot(tm,load)
	plt.show()
	#a=rect_SSL_agent([60*60*2,60*60*24,60*60,1.,0.5/3600.])
	#a.inform_tariff(tf)
	#a.schedule()
	#print a.ts
	#a.plotm()
	#f=open('Adef.txt','w')
	#for i in range(4):
	#	f.write(str([60*60*(i+1),60*60*24,60*60,1.,0.5/3600.])+'\n')
	#f.close()
	#c=community()
	return

main()
