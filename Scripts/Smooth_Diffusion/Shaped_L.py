from matplotlib import cm
import numpy as np
import scipy as sp
import seaborn as sns
import scipy.stats as st
import scipy.linalg as sl
from matplotlib import pyplot as plt
import scipy.integrate as si
from mpl_toolkits import mplot3d


# k = [kprod,kdeg,kdegs,kbind,kunbind,kp,kdp,L]

def react_R(R, B, k, L):
	return k[0] - k[1] * R + k[4] * B - k[3] * L * R


def react_B(R, B, P, k, L):
	return -k[1] * B - k[4] * B + k[3] * L * R - k[5] * B + k[6] * P


def react_P(B, P, k):
	return k[5] * B - k[6] * P - k[2] * P


def L_diff(t, x):
	L0 = 1
	if t == 0:
		return np.zeros(np.size(x))
	else:
		value_list = np.zeros(np.size(x))
		for i in range(np.size(value_list)):
			if i<np.size(value_list)/2:
				value_list[i] = L0
		return value_list
	
def L_diff(t, x,min,max,shift,shift_length):
	if t == 0:
		return np.linspace(min,max,np.size(x))
	else:
		ph_val = np.linspace(min,max,np.size(x))
		ph_val[-shift_length:] += shift
		return np.linspace(0,1,np.size(x))


def step_R(ARinv, dt, react_R, Rn, Bn, k, Ln):
	return np.matmul(ARinv, (Rn + dt * react_R(Rn, Bn, k, Ln)))


def step_B(ABinv, dt, react_B, Rn, Bn, Pn, k, Ln):
	return np.matmul(ABinv, (Bn + dt * react_B(Rn, Bn, Pn, k, Ln)))


def step_P(APinv, dt, react_P, Bn, Pn, k):
	return np.matmul(APinv, (Pn + dt * react_P(Bn, Pn, k)))


def solve(dt, dx, D, L, T, k):
	# k = [kprod,kdeg,kdegs,kbind,kunbind,kp,kdp]
	# Current model assumes R,B,P is in eq with L0
	alpha = D * dt / dx ** 2
	if np.ceil(L / dx) != np.floor(L / dx):
		print(f"Incompatible L and dx,\n L = {L}, dx = {dx}, L/dx = {L / dx}")
	inv_A_list = []
	for a in alpha:
		A = (np.eye(int(L / dx) * 2 + 1) * (1 + 2 * a))
		
		# generates lower left stip of negative alphas below main diagonal
		# index based declaration might make more sense but I believe this is faster
		Abl = (np.tri(int(L / dx) * 2 + 1, k=-1) - np.tri(int(L / dx) * 2 + 1, k=-2)) * -1 * a
		A += (Abl + np.transpose(Abl))
		A[0, 1] -= a
		A[-1, -2] -= a
		Ainv = np.linalg.inv(A)
		inv_A_list.append(Ainv)
	X = np.arange(-L, L + dx, dx)
	
	R_values = np.zeros((T + 1, np.size(X)))
	B_values = np.zeros((T + 1, np.size(X)))
	P_values = np.zeros((T + 1, np.size(X)))
	L_values = np.zeros((T + 1, np.size(X)))
	
	L_values[0] = L_diff(0, X)
	c = k[3] * L_values[0] / (k[1] + k[4] + k[5] - k[5] * k[6] / (k[6] + k[2]))
	
	R_values[0] = k[0] / (k[1] + k[3] * L_values[0] - k[4] * c)
	B_values[0] = c * R_values[0]
	P_values[0] = B_values[0] * k[5] / (k[2] + k[6])
	time_values = np.arange(0, T + 1) * dt
	# [0,dt,2dt,...,T]
	current_time = dt
	for t in range(1, T + 1):
		L_values[t] = L_diff(time_values[t], X)
		R_values[t] = step_R(inv_A_list[0], dt, react_R, R_values[t - 1], B_values[t - 1], k, L_values[t - 1])
		B_values[t] = step_B(inv_A_list[1], dt, react_B, R_values[t - 1], B_values[t - 1], P_values[t - 1], k,
		                     L_values[t - 1])
		P_values[t] = step_P(inv_A_list[2], dt, react_P, B_values[t - 1], P_values[t - 1], k)
		current_time += dt
	return {"R": R_values, "B": B_values, "P": P_values, "L": L_values, "t": time_values}

diffusion_matrix = {'R':0.025,'B':0.025,'P':0.025}

DR = 0.025
DB = 0.025
DP = 0.025
D = np.array([DR,DB,DP])
L = 50
dx = 0.2
dt = 0.5
T_count = 8000
X = np.arange(-L,L+dx,dx)
# k = [kprod,kdeg,kdegs,kbind,kunbind,kp,kdp,L]
kvec = 10**np.array([1.4,-3.6,-2.5,-1.81,-0.52,0.05,-1.15])
# 10**5 = total cell receptor count
kvec[0] /= (np.size(X))
time = [0,100,200,400,600,800]
results = solve(dt,dx,D,L,T_count,kvec)
#plt.plot(results["t"], results["R"][:,0])
#plt.show()


print(np.size(results["t"]))
print(np.size(X))
print(np.shape(results["R"]))
Time, Dime = np.meshgrid(results["t"], X)
list_of_values = ["R","B","P"]
for i in range(len(list_of_values)):
	fig, ax = plt.subplots()
	sns.heatmap(np.transpose(results[list_of_values[i]]), cmap=cm.coolwarm)
	ax.set_title(f"{list_of_values[i]}")
	ax.set_xlabel('Time (m)')
	ax.set_xticks(np.arange(0, np.size(results["t"]), int(T_count/10)))
	ax.set_xticklabels(np.round(dt*results["t"][0::int(T_count/10)]/60, 2))
	
	ax.set_yticks(np.arange(0,np.size(X),100))
	ax.set_yticklabels(np.round(X[0::100],2))
	ax.set_ylabel('X')
	plt.show()