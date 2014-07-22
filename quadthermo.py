#!/usr/bin/env python

import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec

N=1000
M=50
J=20

T0=5.

P=3
cm=4.
k=0.1
dt=1.

Te=sp.matrix([2-2*sp.cos(i*2*sp.pi/float(N)) for i in range(N+1)]).T

Ts=sp.matrix(18*sp.ones([N+1,1]))

Lambda = sp.matrix([4+4*sp.exp(-0.0001*(i-600)**2) for i in range(N+1)]).T

Gamma = sp.matrix(sp.zeros([N+1,1]))
Gamma[0,0]=T0

U=sp.matrix(sp.vstack([sp.zeros([1,N+1]),sp.hstack([sp.eye(N),sp.zeros([N,1])])]))

Phi = P/float(cm) * (sp.eye(N+1)-(1-dt*k/cm)*U).I

Psi=(sp.eye(N+1)-(1-dt*k/cm)*U).I * (dt*k/cm*U*Te+Gamma)

D = sp.vstack([sp.zeros([1,M]),sp.kron(sp.eye(M),sp.ones([J,1]))])

#plt.plot(sp.array( Phi*D*sp.matrix(sp.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])).T + Psi))
#plt.show()

q = sp.matrix(0.1*sp.ones([N+1,1]))
for i in range(250):
	Ts[i,0]=8
	q[i,0]=0.05
for i in range(800,1001):
	Ts[i,0]=8
	q[i,0]=0.05
Q = sp.matrix(sp.zeros([N+1,N+1]))
for i in range(N+1):
	Q[i,i]=q[i,0]


#print D.shape
#print Phi.shape
#print Q.shape
R = D.T*Phi.T*Q*Phi*D
St = 2*(Psi-Ts).T*Q*Phi*D + Lambda.T*D
#print R.shape
#print St.shape
K=sp.matrix(sp.vstack([sp.eye(M),-sp.eye(M)]))
Y=sp.matrix(sp.vstack([sp.ones([M,1]),sp.zeros([M,1])]))
for i in range(100,200):
	e=sp.matrix(sp.zeros([1,N+1]))
	e[0,i]=1
	K=sp.vstack([K,-e*Phi*D])
	Y=sp.vstack([Y,-12+e*Psi])

from cvxopt import matrix,solvers
Q = 2*matrix(R)
p = matrix(sp.array(St).ravel())
G = matrix(K)
h = matrix(sp.array(Y).ravel())

alpha = sp.matrix(sp.zeros([1,N+1]))
for i in range(250,800):
	alpha[0,i]=1
A=alpha*Phi*D
print A.shape
b=sp.matrix(18*(800-250)*sp.ones([1,1]))-alpha*Psi

A = matrix(A)
b = matrix(b)
sol=solvers.qp(Q, p, G, h,A,b)

u=sp.matrix(sol['x'])
T=Phi*D*u+Psi
print sum(T[250:800])/float(800-250)
#print Gamma.T*D*u
plt.figure(figsize=(18,12))

ax0 = plt.subplot2grid((8,5),(0,0),rowspan=4,colspan=5)
ax0.set_ylabel("Temperature")
ax0.plot(sp.array(T))
ax0.plot(sp.array(Ts))
ax0.plot(sp.array(Te))

ax1 = plt.subplot2grid((8,5),(4,0),rowspan=1,colspan=5)
ax1.set_ylabel("Input")
ax1.plot(sp.array(D*u))
ax1.axis([0,N,0,1])

ax2 = plt.subplot2grid((8,5),(5,0),rowspan=1,colspan=5)
ax2.set_ylabel("Tariff")
ax2.plot(sp.array(Lambda))
ax2.axis([0,N,0,10])

ax3 = plt.subplot2grid((8,5),(6,0),rowspan=1,colspan=5)
ax3.set_ylabel("CostWeight")
ax3.plot(sp.array(q))
plt.show()


