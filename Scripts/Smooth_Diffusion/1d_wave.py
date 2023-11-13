import numpy as np
import scipy as sp
import scipy.linalg as sl
from matplotlib import pyplot as plt

def r(u):
	return u-u**2

def u0(x):
	return 0.05*np.exp(-5*x**2)


def step(un,Ainv,dt,ru):
	return np.matmul(Ainv,(un+dt*ru(un)))

def solve(dt,dx,D,L,T,u0):
	alpha = D*dt/dx**2
	if np.ceil(L/dx) != np.floor(L/dx):
		print(f"Incompatible L and dx,\n L = {L}, dx = {dx}, L/dx = {L/dx}")
	A = np.eye(int(L/dx)*2+1)*(1+2*alpha)
	
	# generates lower left stip of negative alphas below main diagonal
	# index based declaration might make more sense but I believe this is faster
	Abl = (np.tri(int(L/dx)*2+1, k=-1) - np.tri(int(L/dx)*2+1, k=-2))*-1*alpha
	A += (Abl + np.transpose(Abl))
	A[0,1] -= alpha
	A[-1,-2] -= alpha
	X = np.arange(-L,L+dx,dx)
	Ainv = np.linalg.inv(A)
	current_t = 0
	un = u0(X)
	for t in range(T):
		un = step(un,Ainv,dt,r)
	return un
	
X = np.arange(-50,50+0.2,0.2)
time = [0,100,200,400,600,800]
for i in time:
	plt.plot(X,solve(0.05,0.2,1,50,i,u0),color = "k")
plt.show()
