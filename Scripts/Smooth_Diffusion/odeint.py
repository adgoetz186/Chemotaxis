import numpy as np
import scipy as sc
import scipy.integrate as si
import scipy.linalg as sl
from matplotlib import pyplot as plt


def linear_diff_eq(t,y,k,L):
	# k = [kprod,kdeg,kdegs,kbind,kunbind,kp,kdp]
	# linear differential equation
	
	sol = np.empty(3)
	
	# dR/dt = -kdeg *[R]-kbind*[L]*[R]+kunbind * [B]+kprod
	sol[0] = -k[1] *y[0]-k[3]*L*y[0]+k[4] * y[1]+k[0]
	
	# dB/dt = -kdeg *[B]+kbind*[L]*[R]-kunbind * [B] + kdp*[P] - kp[B]
	sol[1] = -k[1] * y[1] + k[3] *L* y[0] - k[4] * y[1] + k[6]*y[2] - k[5]*y[1]
	
	# dP/dt = -kdegs *[P]-kdp*[P]+kp * [B]
	sol[2] = -k[2] * y[2] - k[6]*y[2] + k[5]*y[1]
	
	return sol
	
	
y0 = np.array([0,0,0])
kvec = np.array([16,2,3,5,3,6,5])

#k = [kprod,kdeg,kdegs,kbind,kunbind,kp,kdp,L]
#ss_solution
L0 = 0
c = kvec[3]*L0/(kvec[1]+kvec[4]+kvec[5]-kvec[5]*kvec[6]/(kvec[6]+kvec[2]))
R = kvec[0]/(kvec[1]+kvec[3]*L0-kvec[4]*c)
B = c*R
P = B*kvec[5]/(kvec[2]+kvec[6])
ysssol = np.array([R,B,P])
L1 = 10
ysol = si.solve_ivp(linear_diff_eq,(0,0.1),ysssol,args=(kvec,L1))

for i in range(len(ysol.y)):
	plt.plot(ysol.t,ysol.y[i])
	plt.show()