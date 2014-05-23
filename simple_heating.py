#!/usr/bin/env python

import numpy as np
import os
import sys
from matplotlib import pyplot as plt

#heating system for a home to optimise power usage under variable taiff usinghte viterbi alorithm
class heater:
	def __init__(self,constraint,k,theta,timestep,nTpoints):
		self.upper = constraint[1]
		self.lower = constraint[0]
		self.k = k
		self.theta = theta
		self.timestep = timestep
		self.nTpoints=nTpoints
		self.set_Tpoints()
		return

	def set_constraint(self,constraint):
		#user constraints on operation
		self.upper = constraint[1]
		self.lower = constraint[0]
		self.set_Tpoints()
		return
	
	def set_physics(self, k, theta):
		#physics characteristics of the building
		self.k = k
		self.theta = theta
		return

	def set_parameters(self,timestep,nTpoints):
		#operation parameters
		self.timestep = timestep
		self.nTpoints = nTpoints
		self.set_Tpoints()
		return

	def set_Tpoints(self):
		self.Tpoints = list(np.linspace(self.lower,self.upper,self.nTpoints))
		return

	def calc_Tahead(self,T_0,T_ext,on,dt):
		#calculate the temperature at a future time given the current temperature and if heating is on or off
		exf = np.exp(-self.k*dt)
		Tahead = T_ext+(T_0-T_ext)*exf+(1-exf)*on*self.theta/self.k
		return Tahead
	
	def calc_c(self,T_0,T_1,T_ext):
		#calculates the fraction of time that the heating must be on to go from T_0 to T_1 over a timestep
		coef = (T_1-T_ext-(T_0-T_ext-self.theta/self.k)*np.exp(-self.k*self.timestep))*self.k/self.theta
		
		c = 1+(np.log(coef))/(self.k*self.timestep)
		return c

	def calc_T_bounds(self,T_0,T_ext):
		#calculate the range of temperatures that can be reached in a timestep from T_0, upper ifheating is full on, lower if it is off
		ub = self.calc_Tahead(T_0,T_ext,True,self.timestep)
		lb = self.calc_Tahead(T_0,T_ext,False,self.timestep)
		return [lb,ub]

	def detail_T(self):
		#produce high res trace of parameters given the values at each timestep
		res = 1
		times = [i*res for i in range(24*60*60/res)]
		breaktimes = self.profile[0]
		cvals = self.profile[2]
		Tinits = self.profile[1]
		Texts = self.profile[4]
		Tmps = []
		onoff = []
		cv=[]
		T=Tinits[0]
		for t in times:
			on = (cvals[0]*self.timestep+breaktimes[0])>=t
			T=self.calc_Tahead(T,Texts[0],on,res)
			onoff.append(on*1.)
			Tmps.append(T)
			cv.append(cvals[0])
			if t>breaktimes[0]+self.timestep:
				
				breaktimes.pop(0)
				
				cvals.pop(0)
				Tinits.pop(0)
				Texts.pop(0)
				T = Tinits[0]
		self.detT=[times,Tmps,onoff,cv]
		return
				
	def DP(self, costfn, T_ext_fn):
		#compute the response to a given external and tarrif profile
		nt = int(24*60*60/self.timestep)
		times = [(i+1)*self.timestep for i in range(nt-1)]
		time_index=range(nt)
		nT = self.nTpoints
		Temps = self.Tpoints
		Temp_index = range(nT)
		data = []
		init=True
		#for each timestep...
		for t in times[::-1]:
			line = []
			#....for each temperature...
			for T_i0 in Temp_index:
				T_0 = Temps[T_i0]
				[T_lb,T_ub] = self.calc_T_bounds(T_0,T_ext_fn(t))
				best = [10**10,-1,-1]
				#...for each temperature at the next timestep...
				for T_i1 in Temp_index:
					T_1 = Temps[T_i1]
					#...if that temperature can be reached...
					if T_1>T_lb and T_1<T_ub:
						#...find the cost of the transition...
						c = self.calc_c(T_0,T_1,T_ext_fn(t))
						if init:
							#...that's all it it's the first step...
							cost = c*costfn(t)*self.timestep
						else:
							#...otherwise add the state value of the next temperature...
							cost = c*costfn(t)*self.timestep+data[-1][T_i1][0]
							assert cost>data[-1][T_i1][0]
						#print cost
						if cost<best[0]:
							#...and if that's the best so far it becomes the state value of the current temperature...
							best = [cost,T_i1,c]
						
							
				
				line.append(best)
			
			init=False
			
			data.append(line)
		
		#We now have a grid of states in a line but we need to form a loop.
		best_trace = []
		best_close = 10**10
		#for each temperature at the first timestep...
		for ind,T_f in enumerate(data[-1]):
			
			
			trace=[T_f]
			next = T_f[1]
			#...trace back to find which temperature it leads to at the penultimate timestep...
			for j in [-2-i for i in range(nt-2)]:
				trace.append(data[j][next])
				next=data[j][next][1]
			
				
				
			T_m1 = Temps[next]
			T_0 = Temps[ind]
			[T_lb,T_ub] = self.calc_T_bounds(T_m1,T_ext_fn(0.))
			#...if the temperature at the start can be reached from the temperature atthe penultimate step...
			if T_0<T_ub and T_0>T_lb:
				#...find the cost of making that transition...
				close_c = self.calc_c(T_m1,T_0,T_ext_fn(t))
				
				close_cost = close_c*costfn(0.)*self.timestep
				#...and add it to the state value...
				sum_cost = close_cost+T_f[0]
				#...and update if this is the new best value.
				if sum_cost<best_close:
					trace.insert(0,[sum_cost,ind,close_c])
					best_trace=trace
					best_close=sum_cost
		#We now have the list of states for the optimum path
		t_index=[]		
		T_profile=[]
		c_profile=[]
		r_profile=[]
		T_extprofile=[]
		for i,s in enumerate(best_trace):
			#pass through the state list and make arrays of each parameter over time
			t_index.append(i*self.timestep)
			T_profile.append(Temps[best_trace[i-1][1]])
			c_profile.append(s[2])
			r_profile.append(s[2]*costfn(i*self.timestep))
			T_extprofile.append(T_ext_fn(i*self.timestep))
			
		self.profile = [t_index,T_profile,c_profile,r_profile,T_extprofile]
		self.poweruse=sum(c_profile)*self.timestep
		self.day_cost=best_close
		return 

