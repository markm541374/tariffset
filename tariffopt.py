#!/usr/bin/env python

import scipy as sp
import numpy as sp
from matplotlib import pyplot as plt
import operator

#all times are in seconds

class agent:
	def __init__(self):
		#init the agent with its goals
		return
	def inform_tariff(self,T):
		#informs the agent of a new tariff. T should be a function accepting time(s) and returning cost (pounds/W)
		self.T=T
		return
	def get_load_s(self):
		#find the /second power use profile
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
	
	
	def get_load_s(self,t0=0,t1=24*60):
		#assume a 24 hour period from 0
		lp = [0.]*(24*60)
		for i in range(self.ts,self.ts+self.ld):
			lp[i]=self.lv
		return lp

	def evaluate_u(self,dt):
		#find the utility of a given deferal
		if self.t0+dt+self.ld>=self.tf:
			return -sp.Inf
		if dt<0:
			return -sp.Inf
		cost_acc=0
		for i in range(self.ld):
			cost_acc+=self.lv*self.T(self.t0+dt+i)
		u = -cost_acc-dt*self.uf
		
		return u

	def schedule(self):
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

def main():
	tf=lambda x:(4*((x/(24*60*60.))-0.5)**2)/3600.
	a=rect_SSL_agent([60*60*2,60*60*24,60*60,1.,0.5/3600.])
	a.inform_tariff(tf)
	a.schedule()
	print a.ts
	a.plotm()
	return

main()
