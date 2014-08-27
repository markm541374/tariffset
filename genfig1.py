#!/usr/bin/env python

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time
print "this takes about 2 minutes"
def gen_sqexp_k(theta):
	A=theta[0]
	d=sp.matrix(theta[1:]).T
	
	N=sp.matrix(d).shape[0]
	D=sp.eye(N)
	for i in range(N):
		D[i,i]=1./(d[i,0]**2)
	return lambda x,y:A*sp.exp(-0.5*(x-y)*D*(x-y).T)

import tariffopt as TO

e=TO.experiment(gen_sqexp_k,[1,0.2,0.2,0.2,0.2],[[0,1],[-1,1],[-1,1],[-1,1],[-1,1]],["ts1.txt"],[0.4,0.4,0.4,0.4],[0.05,0.05,0.05,0.05],TO.gen_3interp_tariff,TO.load_cost_abflatness)
e.loadtrace('res/trace4.txt')

c,data = e.o.eval_under_tariff(sp.array([0.225,0.225,0.225,0.225]),plot_=True,data_=True)

c,data4=e.o.eval_under_tariff(e.G.best[0],plot_=True,data_=True)
p4=e.G.best[0]


e=TO.experiment(gen_sqexp_k,[1,0.2,0.2,0.2,0.2,0.2,0.2,0.2],[[0,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],["ts1.txt"],[0.4,0.4,0.4,0.4,0.4,0.4,0.4],[0.05,0.05,0.05,0.05,0.05,0.05,0.05],TO.gen_3interp_tariff,TO.load_cost_abflatness)
e.loadtrace('res/trace7.txt')
cs,data7s=e.o.eval_under_tariff(e.G.best[0],plot_=True,data_=True)
p7=e.G.best[0]

pf=plt.figure(figsize=(18,12))
ax0 = plt.subplot2grid((8,5),(3,0),rowspan=1,colspan=5)
ax1 = plt.subplot2grid((8,5),(0,0),rowspan=3,colspan=5)

ax0.plot(data[0],'r')
ax0.plot(data4[0],'g')
ax0.plot(data7[0],'b')
M=lambda x:x/1000000.
ax1.plot(map(M,data[1]),'r')
ax1.plot(map(M,data4[1]),'g')
ax1.plot(map(M,data7[1]),'b')

spoints4 = [i*24*60/float(4-1) for i in range(4)]
spoints7 = [i*24*60/float(7-1) for i in range(7)]

ax0.plot(spoints4,100*p4,'go')
ax0.plot(spoints7,100*p7,'bo')

ax0.axis([0,1440,0,50])

ax1.axis([0,1440,0,5])
ax1.get_xaxis().set_visible(False)
ax0.set_xlabel('Time (minutes)')
ax0.set_ylabel('Tariff (pencekWh)')
ax1.set_ylabel('Load (MW)')

pf.show()
