#!/usr/bin/env python

import scipy as sp
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time

x=[4,5,6,7,8,9]
ytr=[0.330225609958,0.322842918383,0.3144159231,0.309315318925,0.31154719201,0.309285150194]
yte=[0.32375780099,0.307815159221,0.309704193136,0.301660757516,0.299501258971,0.294388329211]
reftr=0.353884216251
refte=0.355785992762


plt.figure(figsize=([5,5]))
plt.plot(x,ytr,'bo-')
plt.plot(x,yte,'ro-')
plt.plot([1,10],[reftr,reftr],'b')
plt.plot([1,10],[refte,refte],'r')
plt.axis([3,10,0.2,0.4])
plt.xlabel('Tariff degrees of freedom')
plt.ylabel('Normalised deviation from mean load')
#plt.show()
plt.savefig('f2.eps')
