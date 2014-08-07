#!/usr/bin/env python

import scipy as sp
import scipy.linalg as spl
import matplotlib.pyplot as plt
import sys
import os
import ast

def readfile(fname):

	X=[]
	Y=[]
	for l in open(fname):
		[x,y]= l[1:].strip(']\n').split(',')
		X.append(map(float,x[2:].strip(']]').split()))
		Y.append([float(y)])

	X=sp.matrix(X)
	Y=sp.matrix(Y)
	return X,Y

def savetrace(X,Y,fname):
        sp.savetxt(fname,sp.hstack([X,Y]))
        return

X,Y=readfile('spares_81/abflatness.txt')
savetrace(X,Y,'spares_81/abtrace.txt')
