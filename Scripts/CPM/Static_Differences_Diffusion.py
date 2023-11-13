import numpy as np
import copy as cp
import cython
import matplotlib.pyplot as plt
import time

import pysb.integrate

from Scripts.Models.Simple_Receptor import model
from pysb.simulator import BngSimulator
#import tensorflow as tf
import pandas as pd

print(np.ones((1000,1000)))

def perimeter(cell_coords):
	pari = 0
	for i in cell_coords:
		nc = np.sum(np.abs(i - cell_coords), axis=1)
		pari += (4 - np.count_nonzero(nc == 1))
	return pari

def perimeter_change(prev_cell_coords,new_coord):
	nc = np.sum(np.abs(new_coord - prev_cell_coords),axis=1)
	return 4-np.count_nonzero(nc == 1)

def generate_internal_external(cell_coords):
	proposed_points = np.zeros((4,2))
	border_points_empty = []
	border_points_filled = []
	for i in cell_coords:
		proposed_points[0] = [i[0]+1,i[1]]
		proposed_points[1] = [i[0]-1,i[1]]
		proposed_points[2] = [i[0], i[1]+1]
		proposed_points[3] = [i[0], i[1]-1]
		nc = np.sum(np.abs(i - cell_coords), axis=1)
		if np.count_nonzero(nc == 1) < 4:
			border_points_filled.append(i)
		for i in range(4):
			if not (np.any(np.equal(cell_coords, (proposed_points[i])).all(1))):
				border_points_empty.append(cp.copy(proposed_points[i]))
	border_points_empty = np.unique(np.array(border_points_empty),axis = 0)
	border_points_filled = np.unique(np.array(border_points_filled), axis=0)
	return [border_points_empty,border_points_filled]

def single_copy_sim_bp(cell_coords,border_points_filled,border_points_empty):
	point_val = np.random.randint(np.shape(border_points_filled)[0] + np.shape(border_points_empty)[0])
	if point_val > (np.shape(border_points_filled)[0]-1):
		point_val = point_val - np.shape(border_points_filled)[0]
		point = border_points_empty[point_val]
		
		rand_val = np.random.randint(4)
		# selects location to copy state to
		change_state_loc = np.array((1 - (2 * (rand_val % 2))) * np.array([1 - rand_val // 2, rand_val // 2]))
		new_state = point + change_state_loc
		
		# checks to see if the selected point contains a cell component and if it does removes it
		indexing = np.where(np.all(cell_coords == new_state, axis=1))
		if len(indexing[0]) == 1:
			cell_coords = np.delete(cell_coords,indexing[0][0],axis=0)
	else:
		print('internal')

def display_cell(cell_coords,xrange = None, yrange = None):
	if xrange is None:
		xrange = [-10,10]
	if yrange is None:
		yrange = [-10,10]
	canvas = np.zeros((xrange[1]-xrange[0],yrange[1]-yrange[0]))
	for i in cell_coords:
		if i[0] >= xrange[0] and i[0] <= xrange[1]:
			if i[1] >= yrange[0] and i[1] <= yrange[1]:
				canvas[i[0]-xrange[0],i[1]-yrange[0]] += 1
	fig,ax = plt.subplots(1)
	ax.imshow(canvas)
	ax.invert_yaxis()
	ax.set_xticklabels(ax.get_xticks()+xrange[0])
	ax.set_yticklabels(ax.get_yticks() + yrange[0])
	ax.set_title(f"Area = {np.shape(cell_coords)[0]}\nCOM = {np.round(np.average(cell_coords,axis=0),2)}")
	plt.show()

def energy_from_cell_coords(cell_coords,A0,lambda_val,alpha):
	A = np.shape(cell_coords)[0]
	L = perimeter(cell_coords)
	return L*alpha + lambda_val*(A0-A)**2

def add_new_space(cell_points_no_POI,POI, species_levels):
	#print("add")
	index_vals = []
	min_val = np.min(np.sum(np.abs(cell_points_no_POI - POI),axis=1))
	if min_val != 1:
		print("ADD: larger than normal distance, distance is: ",min_val)
	#print(min_val)
	for i in range(np.shape(cell_points_no_POI)[0]):
		if np.sum(np.abs(POI - cell_points_no_POI[i])) == min_val:
			index_vals.append(i)
	update_val = np.sum(species_levels[index_vals,:],axis=0)/(len(index_vals)+1)
	species_levels[index_vals,:] = update_val
	species_levels = np.vstack((species_levels,update_val))
	cell_points = np.vstack((cell_points_no_POI, POI))
	if np.isnan(species_levels).any():
		print("Grow")
		print("ind: ", len(index_vals))
		print("update: ", update_val)
		print("zero_ind: ", zero_ind)
		print("Cell Points: ", cell_points_with_POI)
		print("species_levels: ", species_levels)
	#print(cell_points)
	return cell_points,species_levels


def remove_space(cell_points_with_POI,POI, species_levels):
	#print("remove")
	index_vals = []
	if np.shape(cell_points_with_POI)[0] == 1:
		print("LOOK")
		input()
	min_val = np.min(np.sort(np.sum(np.abs(cell_points_with_POI - POI), axis=1))[1:])
	if min_val != 1:
		print("REM: larger than normal distance, distance is: ",min_val)
	#print(min_val)
	for i in range(np.shape(cell_points_with_POI)[0]):
		if np.sum(np.abs(POI - cell_points_with_POI[i])) == min_val:
			index_vals.append(i)
	zero_ind = np.argmin(np.sum(np.abs(cell_points_with_POI - POI), axis=1))
	update_val = species_levels[zero_ind]/len(index_vals)
	if np.isnan(species_levels).any():
		print("ind: ", len(index_vals))
		print("update: ",update_val)
		print("zero_ind: ",zero_ind)
		print("Cell Points: ",cell_points_with_POI)
		print("species_levels: ", species_levels)
	species_levels[index_vals,:] += update_val
	species_levels = np.delete(species_levels,zero_ind,axis=0)
	cell_points = np.delete(cell_points_with_POI,zero_ind,axis=0)
	if np.isnan(species_levels).any():
		print("cell points 2: ", cell_points)
		print("species_levels 2: ", species_levels)
	if np.shape(cell_points_with_POI)[0] == 1:
		print(cell_points)
		input()
	return cell_points,species_levels
			
			

def single_copy_sim(cell_dtf,A0 = 100,lambda_val = 0.01,alpha = 2,buffer = 5):
	# PLACEHOLDER
	cell_coords = cell_dtf[["x_val", "y_val"]].to_numpy()
	species_list = cell_dtf.columns.to_list()
	species_list.remove("x_val")
	species_list.remove("y_val")
	species_levels = cell_dtf[species_list].to_numpy()
	starting_level = np.sum(species_levels,axis=0)
	canvas_size = np.max(cell_coords,axis=0)+1 - (np.min(cell_coords,axis=0))+buffer

	canvas_size = int(canvas_size[0]*canvas_size[1])
	# add some form of check to see if the buffer is too small
	points = np.random.randint(np.min(cell_coords, axis=0) - 1, np.max(cell_coords, axis=0) + 2, (canvas_size, 2))
	size_traj = []
	COMx_traj = []
	COMy_traj = []
	point_count = 0
	p = np.array([0.0,0.0])
	X_prev = np.array([0,0])
	eta = 107
	r = 0.01*0
	epsilon = 56
	tau = 100
	for point_val in points:

		#print(point_val)
		#print(cell_coords)
		if np.shape(cell_coords)[0] != 0:
			# is the selected point a cell
			indexing_orig = np.where(np.all(cell_coords == point_val, axis=1))
			rand_val = np.random.randint(4)
			# selects location to copy state to
			change_state_loc = np.array((1 - (2 * (rand_val % 2))) * np.array([1 - rand_val // 2, rand_val // 2]))
			change_state_loc += point_val
			indexing_new = np.where(np.all(cell_coords == change_state_loc, axis=1))
			if len(indexing_new[0]) != len(indexing_orig[0]):
				#display_cell(cell_coords,xrange=[-50,50],yrange=[-50,50])
				if len(indexing_orig[0]) == 1:
					# the cell is attempting to grow
					U_init = energy_from_cell_coords(cell_coords,A0,lambda_val,alpha)
					cell_coords_proposed = np.vstack((cell_coords,change_state_loc))
					U_proposed = energy_from_cell_coords(cell_coords_proposed, A0, lambda_val, alpha)
					dX = np.average(cell_coords_proposed,axis=0) - np.average(cell_coords,axis=0)

					#print(np.reshape(species_levels[:,-1],(-1,1)))
					q = np.average((cell_coords - np.average(cell_coords, axis=0))*np.reshape(species_levels[:,-1],(-1,1)),axis=0)
					# Where to do the detection step. Do you update each time or make the choices then do the timestep
					#print(tau*r*(-p+eta*X_prev+epsilon*q))
					p += tau*r*(-p+eta*X_prev+epsilon*q)
					w = np.dot(dX,p)
					#print(w)
					if U_proposed-U_init - w < 0:
						cell_coords, species_levels = add_new_space(cell_coords, change_state_loc, species_levels)
						X_prev = dX
					else:
						prob = -(U_proposed-U_init - w)
						if np.log(np.random.random(1)) < prob:
							cell_coords, species_levels = add_new_space(cell_coords, change_state_loc, species_levels)
							X_prev = dX
				else:
					# the cell is attempting to shrink
					U_init = energy_from_cell_coords(cell_coords, A0, lambda_val, alpha)
					cell_coords_proposed = np.delete(cell_coords,indexing_new[0][0],axis=0)
					U_proposed = energy_from_cell_coords(cell_coords_proposed, A0, lambda_val, alpha)
					dX = np.average(cell_coords_proposed, axis=0) - np.average(cell_coords, axis=0)

					q = np.average(
						(cell_coords - np.average(cell_coords, axis=0)) * np.reshape(species_levels[:, -1], (-1, 1)),
						axis=0)
					p += tau * r * (-p + eta * X_prev + epsilon * q)
					w = np.dot(dX, p)
					if U_proposed - U_init - w < 0:
						cell_coords, species_levels = remove_space(cell_coords, change_state_loc, species_levels)
						X_prev = dX
					else:
						prob = -(U_proposed-U_init - w)
						if np.log(np.random.random(1)) < prob:
							cell_coords, species_levels = remove_space(cell_coords, change_state_loc, species_levels)
							X_prev = dX
					#input()
				size_traj.append(np.shape(cell_coords)[0])
				COMx_traj.append(np.average(cell_coords[:,0]))
				COMy_traj.append(np.average(cell_coords[:, 1]))
				point_count+=1
		else:
			size_traj.append(np.shape(cell_coords)[0])
			COMx_traj.append(np.average(cell_coords[:, 0]))
			COMy_traj.append(np.average(cell_coords[:, 1]))
			point_count += 1
	cell_dtf = pd.DataFrame.from_records(np.hstack((cell_coords,species_levels)), columns=cell_dtf.columns.to_list())
	if np.sum(np.abs(starting_level - np.sum(species_levels, axis=0))) > 1e-9:
		print((np.abs(starting_level - np.sum(species_levels, axis=0))))
		print("WARNING! CELL MOVEMENT HAS CAUSED SPECIES CHANGE")
	return cell_dtf

def sim_point(cell_dtf,coord,sim_obj,param_dict,grad_drop,grad_base):
	# updates a single cells values using a deterministic solution
	A = time.time()
	point_df = cell_dtf.loc[(cell_dtf.x_val==coord[0]) & (cell_dtf.y_val==coord[1])]
	point_df = point_df.rename(columns={"R": "R_0", "B": "B_0", "P": "P_0"})
	dict_val = dict(zip(["R_0", "B_0", "P_0"],point_df[["R_0", "B_0", "P_0"]].to_numpy()[0].tolist()))
	param_dict.update(dict_val)
	param_dict["kprod"] /= cell_dtf.shape[0]
	param_dict["EGF"] = coord[0]*grad_drop+grad_base
	#param_dict["EGF"] = 0
	#print(param_dict["kprod"],cell_dtf.shape[0], 1)
	#param_dict.update(point_df[["R_0","B_0","P_0"]])
	B = time.time()
	#print(param_dict.values())


	sim_obj.run(param_values=param_dict)

	#print(param_dict)
	C = time.time()
	#x = sim.run(n_runs=1, method='ode')
	#y = ScipyOdeSimulator(model,tspan = t_vec, param_values = param_dict)
	D = time.time()
	#results = x.dataframe[["obs_R","obs_B","obs_P"]].iloc[-1]
	cell_dtf.loc[(cell_dtf.x_val == coord[0]) & (cell_dtf.y_val == coord[1]),["R","B","P"]] = sim_obj.y[-1,:]
	E = time.time()
	param_dict["kprod"] *= cell_dtf.shape[0]
	#print(param_dict["kprod"],2)
	return np.array([B-A,C-B,D-C,E-D])
	#results = results[["R_0", "B_0", 'P_0']].iloc[-1]
	
def sim_points(cell_dtf,sim_obj,param_dict,grad_drop,grad_base):
	clocks = np.zeros(4)
	xy_points = cell_dtf[["x_val","y_val"]].to_numpy()
	for point in xy_points:
		clocks += sim_point(cell_dtf,point,sim_obj,cp.copy(param_dict),grad_drop,grad_base)




def generate_connection_matrix(cell_dtf):
	# convert to np array
	cell_coords = cell_dtf[["x_val","y_val"]].to_numpy()
	transition_matrix = np.zeros((np.shape(cell_coords)[0],np.shape(cell_coords)[0]))
	for point_ind_1 in range(np.shape(cell_coords)[0]):
		for point_ind_2 in range(np.shape(cell_coords)[0]):
			if np.sum(np.abs(cell_coords[point_ind_1] - cell_coords[point_ind_2])) == 1:
				transition_matrix[point_ind_1,point_ind_2] = 1
		if np.sum(transition_matrix[point_ind_1] ) == 0:
			transition_matrix[point_ind_1,point_ind_1] = 1
		else:
			transition_matrix[point_ind_1] /= np.sum(transition_matrix[point_ind_1] )
	return transition_matrix
			

def diffusion_step(cell_dtf,diffusion_dict):
	# generalize this to all species
	connection_matrix = generate_connection_matrix(cell_dtf)
	for species in diffusion_dict.keys():
		species_diffusion_mat = connection_matrix*diffusion_dict[species]
		for point_ind_1 in range(np.shape(species_diffusion_mat)[0]):
			species_diffusion_mat[point_ind_1,point_ind_1] = 1-diffusion_dict[species]
		new_vec = np.matmul(cell_dtf[species].to_numpy(),species_diffusion_mat)
		cell_dtf[species] = new_vec
		if np.isnan(new_vec).any():
			print("error")
			print(species_diffusion_mat)
			print(new_vec)
			print(connection_matrix)
		

	

# note calculation of area is just np.shape(cell_coords)[0]

R0 = 300000
L = [0,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,100]
D = 0

grad_dist = 1300
units_of_grid = 2
grad_conc = 50
print(grad_conc/grad_dist/10)
print(np.log10(grad_conc/grad_dist/10))

grad_drop = grad_conc/grad_dist*units_of_grid/100
grad_base = 0.125
grad_bases = [0,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1]
min_val = []
max_val = []
dist_val = []
for grad_base in grad_bases:
	print(grad_drop)
	
	
	
	# natural surface area of cell
	A0 = 100
	
	# maximum surface area of the cell
	Amax = 20
	
	# fluctuation energy cost
	l = 2
	
	# Adhesion energy
	alpha = 15
	
	initial_cell = np.array([[0,0],[0,1],[1,0]])
	exterior_filled = []
	exterior_empty = []
	
	
	#display_cell(initial_cell)
	diffusion_matrix = {'R':0.025,'B':0.025,'P':0.025}
	x_vals = []
	y_vals = []
	for i in range(10):
		x_vals.append(i)
		y_vals.append(0)
	Cell_df = pd.DataFrame({'x_val':x_vals,'y_val':y_vals,"R":[0 for i in range(len(x_vals))],"B":[0 for i in range(len(x_vals))],"P":[0 for i in range(len(x_vals))]})
	t_main = np.linspace(0,100,10)
	dtf_dict = {'kprod':0.05876103899761891/0.00122,'kbind':0.01625578123538074,'kunbind': 0.39307675090835553,'krp':0.4777926053914779,'krdp':0.04702130222751365,'kdeg':0.000329582245342056,'ksdeg':0.0010486403956587256,'R_0':0,'B_0':0,'P_0':0,'EGF':1}
	#print(Cell_df)
	#print(dtf_dict['kprod']/dtf_dict['kdeg'])
	
	AZ = time.time()
	sims = 0
	sim_obj = pysb.integrate.Solver(model, tspan=t_main)
	# 400
	for repeat in range(400):
		print(repeat,grad_base)
		Z = time.time()
		#Cell_df = single_copy_sim(Cell_df)
		A = time.time()
		sim_points(Cell_df,sim_obj,dtf_dict,grad_drop,grad_base)
		B = time.time()
		diffusion_step(Cell_df,diffusion_matrix)
		sims += Cell_df.shape[0]
		C = time.time()
		#print(Cell_df.shape[0])
		#print(np.sum(Cell_df[["R"]].to_numpy()))
		#print(np.average(Cell_df[["x_val","y_val"]].to_numpy(),axis=0))
		
		#print((A-Z)/(C-Z),(B-A)/(C-Z),(C-B)/(C-Z))
		#print((-1*(AZ-time.time())/(sims)))
	min_val.append(Cell_df[["P"]].to_numpy()[0])
	max_val.append(Cell_df[["P"]].to_numpy()[-1])
	dist_val.append((Cell_df[["P"]].to_numpy()[-1]-Cell_df[["P"]].to_numpy()[0])/np.sqrt(Cell_df[["P"]].to_numpy()[-1]))
plt_values = np.array(cp.copy(grad_bases))
plt_values[0] = 0.001
plt.scatter(np.log10(plt_values),min_val)
plt.scatter(np.log10(plt_values),max_val)
plt.ylabel("P")
plt.xlabel(f"Baseline ng/mL (log)")
plt.title(f"gradient: {np.round(np.log10(grad_drop/units_of_grid),2)} ng/mL/$\mu$m")
plt.show()
plt.ylabel("Range normalized by sqrt max")
plt.xlabel(f"Baseline ng/mL (log)")
plt.title(f"gradient: {np.round(np.log10(grad_drop/units_of_grid),2)} ng/mL/$\mu$m")
plt.scatter(np.log10(plt_values),dist_val)
plt.show()
#print(dtf_dict)
print(time.time()-AZ)
input()
sim = BngSimulator(model, tspan=t_main,param_values=dtf_dict)
print(dtf_dict)

# This initializes the model it is not a bottleneck, but could be made more efficient
print(model.species)
x = sim.run(n_runs=1, method='ode')
print(x.dataframe)
print(x.species)
results = x.dataframe.rename(columns={"obs_R": "R_0", "obs_B": "B_0","obs_P":"P_0"})
results = results[["R_0","B_0",'P_0']].iloc[-1]

print(results)
dtf_dict.update(results)
sim = BngSimulator(model, tspan=t_main,param_values=dtf_dict)
print(dtf_dict)
# This initializes the model it is not a bottleneck, but could be made more efficient
print(model.species)
x = sim.run(n_runs=1, method='ode')
print(dtf_dict)
print(x.dataframe)
input()
#x = sim.run(n_runs=1, method='ode', initials=initial_row[1:4])
#x = sim.run(n_runs=1, method='pla',initials = state)
print(Cell_df)
input()
#border_points = generate_internal_external(initial_cell)
#print(border_points)
traj = np.zeros((1,16000))
xcom = np.zeros((1,16000))
ycom = np.zeros((1,16000))
start = time.time()
for i in range(400):
	#single_copy_sim_bp(initial_cell,border_points[1],border_points[0])
	initial_cell = single_copy_sim(initial_cell)
	print(i)
	print(np.shape(initial_cell)[0])
print(time.time()-start)
input()
# Make sure you are able to handle situations where single pixel dissapears maybe migrate to nearest cell
plt.plot(np.arange(16000),np.average(xcom,axis=0))
plt.plot(np.arange(16000),np.average(ycom,axis=0))
plt.show()
plt.plot(np.arange(16000),np.average(traj,axis=0))
plt.show()
vec_size = 100
vec = np.zeros(vec_size)
transport_vec = np.eye(vec_size-1)
print(transport_vec)
transport_vec_low = np.vstack((np.zeros(vec_size-1),transport_vec))
transport_vec_low = np.hstack((transport_vec_low,np.zeros((vec_size,1))))
transport_vec_low = (transport_vec_low + transport_vec_low.T)/2
vec[3] = 1
print(transport_vec_low)
transport_vec_low[0,1] = 0
transport_vec_low[0,0] = 1
print(transport_vec_low)
print(np.matmul(vec,transport_vec_low))

# Cell square database


for i in range(160):
	plt.bar(np.arange(vec_size-1)-.25,np.histogram(traj[:, i], bins=np.arange(vec_size))[0] / 10000,width=.5)
	vec = np.matmul(vec,transport_vec_low)
	print(i)
	plt.bar(np.arange(vec_size-1)+.25,vec[:-1],width=.5)
	plt.show()
plt.plot(np.arange(16),np.average(traj,axis=0))
plt.show()