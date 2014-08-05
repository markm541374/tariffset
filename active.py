#!/usr/bin/env python
import scipy as sp
import matplotlib.pyplot as plt

def readtable(n,we):
	if we:
		fname='occtables/active'+str(n)+'e.txt'
	else:
		fname='occtables/active'+str(n)+'d.txt'
	raw = [i.strip('\n').split('\t') for i in open(fname)]
	table = sp.zeros([144,7,7])
	for e in raw:
		i=int(e[0])-1
		j=int(e[1])
		for k,p in enumerate(e[2:]):
			table[i,j,k]=float(p)
	return table

def n10m2sst(D):
	#convert number of occupants on 10min vector to startstop times
	state=0
	times=[]
	for i,e in enumerate(D):
		if e and not state:
			state=1
			newstart=i*10
		if e and state:
			pass
		if not e and state:
			state=0
			newstop=i*10
			times.append([newstart,newstop])
		if not e and not state:
			pass
	if state:

		newstop=i*10
		times.append([newstart,newstop])
	return times

#raw=[int(i.strip('\n')) for i in open('tmp.txt')]
#print n10m2sst(raw)

def init(n,we=0):
	if we==0:
		table=sp.array([[0.844,0.798,0.754,0.734,0.680,0.615],[0.156,0.145,0.189,0.182,0.176,0.205],[0.000,0.058,0.044,0.061,0.120,0.103],[0.000,0.000,0.013,0.020,0.008,0.051],[0.000,0.000,0.000,0.002,0.008,0.000],[0.000,0.000,0.000,0.000,0.008,0.026],[0.000,0.000,0.000,0.000,0.000,0.000]])
	if we==1:
		table=sp.array([[0.829,0.768,0.676,0.676,0.547,0.667],[0.171,0.161,0.210,0.207,0.250,0.139],[0.000,0.071,0.094,0.092,0.109,0.111],[0.000,0.000,0.019,0.016,0.078,0.028],[0.000,0.000,0.000,0.009,0.000,0.056],[0.000,0.000,0.000,0.000,0.016,0.000],[0.000,0.000,0.000,0.000,0.000,0.000]])
	vec=table[:,n-1]
	r=sp.random.uniform()
	i=0
	while r>vec[i]:
		r-=vec[i]
		i+=1
	return i

def runmc(n,we=0):
	if n>6:
		print "warning, n>6, rounded down to 6"
		n=6
	table=readtable(n,we)
	
	state=init(n,we)
	occ=[state]
	for tstep in table:
		vec=tstep[state]
		
		r=sp.random.uniform(0,1)
		i=0
		while i<7 and r>vec[i] :
			r-=vec[i]
			i+=1
		state=i
		occ.append(state)
	return occ

def get_occ_p(n,we=0):
	return n10m2sst(runmc(n,we))
