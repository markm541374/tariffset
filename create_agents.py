#!/usr/bin/env python

from scipy import random as spr
import scipy as sp
import active

def rect_SSL_agent_gen():
	t0=int(min(3600*spr.gamma(6,1),23*3600)) #time the agent can start
	tf=int(min(t0+3600*(1+spr.gamma(12,1)),24*3600)) #time the agent must finish by (s)
	ld=3600 #time the agent runs for (s)
	lv= 1.#value of the load when running
	uf=0.1*spr.gamma(1,1)/3600. #utility gradient for deferal
	s=str(["rect_SSL_agent",t0,tf,ld,lv,uf])
	return s

def quadthermo_agent_gen():
	targetT=spr.normal(18,2)
	tolerance=1.5

	absmax=max(spr.normal(21,1),targetT+tolerance) #no support
	absmin=min(spr.gamma(4,1),targetT-tolerance) #no support

	vec=[0.0,0.21,0.135,0.205,0.115,0.115,0.095,0.065,0.03,0.03]
	r=sp.random.uniform()
	nres=1
	i=0
	
	while r>vec[i]:
		r-=vec[i]
		i+=1
	
		
	nres=i+1
	
	occ=active.get_occ_p(nres)
	cons=[]
	for p in occ:
		p.extend([targetT,tolerance])
		cons.append(p)
	q=0.005 #magic number
	
	
	fa=28.39+sp.random.gamma(shape=2.099,scale=28.696) #floor area from cabe dwelling survey
	
	flat=sp.random.uniform(0,1)<0.365 # flat or house
	if flat:
		U = 3.3*sp.sqrt(fa) #insulation in W/K floor area
	else:
		U= 3.6*sp.sqrt(fa)+0.14*fa
	
	
	k=U #insulation in W/K

	cm=1000*sp.exp(sp.random.normal(5.5,0.35)) #thermal capacity in J
	
	P=sp.random.uniform(6000,15000)# power in W

	Prequ=k*20
	
	if Prequ>P:
		print "!!!!!!!!!!!"
	s=str(["quadthermo_agent",absmax,absmin,cons,q,P,cm,k])
	return s

def writeagents(fname,nqth,nssl):
	f=open(fname,'w+')
	for i in range(nqth):
		f.write(quadthermo_agent_gen()+'\n')
	for i in range(nssl):
		f.write(rect_SSL_agent_gen()+'\n')
	f.close()
	return

writeagents('ts0.txt',20,0)
writeagents('ts1.txt',20,0)
writeagents('ts2.txt',20,0)
