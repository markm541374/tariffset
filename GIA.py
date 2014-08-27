#!/usr/bin/env python
import scipy as sp
from scipy import random as spr
from scipy import stats as spt
from matplotlib import pyplot as plt
import operator


#this takes the data on gross internal area of 200 homes from 
#http://data.gov.uk/dataset/floor-space-data-for-english-houses-and-flats-may-2010
#and fitts an offset gamma distribution
#outputs
#loc = 28.39
#shape = 2.09878208509
#scale = 28.6956423099


data = [float(i.strip('\n')) for i in open("GIA.txt")]
N=len(data)


mean = sum(data)/float(N)
var = sum([((i-mean)**2)/float(N) for i in data])

scale=var/mean
shape=mean/scale

hp,bins,patches=plt.hist(data,bins=30,normed=1,facecolor='green',alpha=0.4)
cbins=[i-0.5*(bins[1]-bins[0]) for i in bins[1:]]

best=sp.Inf
jmax=0
for j in sp.linspace(0,min(data),20):
	adjmean=mean-j
	scale=var/adjmean
	shape=adjmean/scale
	rv = spt.gamma(shape,scale=scale,loc=j)

	approx = [rv.pdf(i) for i in cbins]
	delta = map(operator.sub,hp,approx)
	u = sum([i**2 for i in delta])
	if u<best:
		jmax=[rv,shape,scale,j]
	

plt.plot(sp.linspace(0,max(data),100),[jmax[0].pdf(i) for i in sp.linspace(0,max(data),100)],'r')
print "loc = "+str(jmax[3])
print "shape = "+str(jmax[1])
print "scale = "+str(jmax[2])
plt.xlabel("Internal Area (m^2)")
plt.ylabel("p(A)")
plt.savefig("f3.eps")
plt.show()


