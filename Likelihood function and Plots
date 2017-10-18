import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
from mpl_toolkits.mplot3d import axes3d


#generate parabola
x=np.linspace(-10,9,1000)
y=2*(x**2) + (4*x) - 5

#add noise
noise = np.random.normal(0,0.5,len(x))
xnew=x+noise

#parameters to iter over
a = np.arange(-3,4,0.2)
b = np.arange(-6,8,0.4)
c = np.arange(-7,7,0.4)
sig=np.arange(0.5,5.5,0.5)

#LIKELIHOOD FUNCTION
def lbit(a,b,c,q,y,sig):
    xdos=np.zeros(len(q))
    for l in range(0,len(q)):
        xdos[l]=-((a*(q[l]**2) + (b*q[l]) + c - y[l])**2)/(2.*sig*sig)
        xdos[l]= xdos[l] - np.log((np.sqrt(2*np.pi*sig*sig)))
    p = np.sum(xdos)
    #pdb.set_trace()
    return p

L=np.zeros((len(a),len(b),len(c)))
for k in range(len(a)):
    print (k)
    for h in range(len(b)):
        for j in range(len(c)):
            L[k,h,j]=lbit(a[k],b[h],c[j],xnew,y,0.5)

#pdb.set_trace()

#marginal distributions
lab=np.zeros((len(a),len(b)))
for i in range(len(a)):
    for j in range(len(b)):
        lab[i,j]=np.sum((L[i,j,:]))

lac=np.zeros((len(a),len(c)))
for i in range(len(a)):
	for j in range(len(c)):
		lac[i,j]=np.sum((L[i,:,j]))

lbc=np.zeros((len(b),len(c)))
for i in range(len(b)):
	for j in range(len(c)):
		lbc[i,j]=np.sum((L[:,i,j]))
####

la_b=np.zeros(len(a))
for i in range(len(a)):
	la_b[i]=np.sum((lab[i,:]))

la_c=np.zeros(len(a))
for i in range(len(a)):
	la_c[i]=np.sum((lac[i,:]))

lb_a=np.zeros(len(b))
for i in range(len(b)):
	lb_a[i]=np.sum((lab[:,i]))

lb_c=np.zeros(len(b))
for i in range(len(b)):
	lb_c[i]=np.sum((lbc[i,:]))

lc_b=np.zeros(len(c))
for i in range(len(c)):
	lc_b[i]=np.sum((lbc[:,i]))

lc_a=np.zeros(len(c))
for i in range(len(c)):
	lc_a[i]=np.sum((lac[:,i]))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.imshow((lab), interpolation='none', origin='lower',extent=[min(a),max(a),min(b),max(b)],aspect='auto')
plt.xlabel('a: True = 2')
plt.ylabel('b: True = 4')
plt.colorbar()
plt.show()


plt.plot(xnew,a[25]*(xnew**2) + (b[25]*xnew)+c[5],label='MLE:a=1.6,b=-5.2,c=-0.2')
plt.plot(xnew,y,label='True:a=2,b=4,c=-5')
plt.legend()
plt.show()

A,B=np.meshgrid(a,b)
B,C=np.meshgrid(b,c)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(A,B,lab, rstride=8, cstride=8, alpha=0.3)


ax.plot(a, la_b, 'r+', zdir='x')
ax.plot(b, lb_a, 'g+', zdir='y')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('L')

plt.show()


plt.plot(c,lc_a,'o-',label='Summed over a')
plt.plot(c,lc_b,label='Summed over b')
plt.xlabel('c')
plt.ylabel('L')
plt.legend(loc=4)
plt.show()
