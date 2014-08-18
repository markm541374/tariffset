#!/usr/bin/env python

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time

def gen_sqexp_k(theta):
	A=theta[0]
	d=sp.matrix(theta[1:]).T
	
	N=sp.matrix(d).shape[0]
	D=sp.eye(N)
	for i in range(N):
		D[i,i]=1./(d[i,0]**2)
	return lambda x,y:A*sp.exp(-0.5*(x-y)*D*(x-y).T)

import tariffopt as TO
e=TO.experiment(gen_sqexp_k,[1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],[[0,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]],["ts1.txt"],[0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],TO.gen_3interp_tariff,TO.load_cost_abflatness)
e.stepn(100)

e.G.savetrace('res/trace8.txt')
e.G.savehyptrace("res/hyptrace8.txt")
