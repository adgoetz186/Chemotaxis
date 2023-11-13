import numpy as np
import copy as cp
import cython
import matplotlib.pyplot as plt
import time
import sys
import os
from pathlib import Path

import pysb.integrate

from Scripts.Models.Simple_Receptor import model
from pysb.simulator import BngSimulator
#import tensorflow as tf
import pandas as pd

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
	index_vals = []
	min_val = np.min(np.sum(np.abs(cell_points_no_POI - POI),axis=1))
	if min_val != 1:
		print("ADD: larger than normal distance, distance is: ",min_val)
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

	return cell_points,species_levels


def remove_space(cell_points_with_POI,POI, species_levels):
	index_vals = []
	min_val = np.min(np.sort(np.sum(np.abs(cell_points_with_POI - POI), axis=1))[1:])
	if min_val != 1:
		print("REM: larger than normal distance, distance is: ",min_val)
	for i in range(np.shape(cell_points_with_POI)[0]):
		if np.sum(np.abs(POI - cell_points_with_POI[i])) == min_val:
			index_vals.append(i)
	zero_ind = np.argmin(np.sum(np.abs(cell_points_with_POI - POI), axis=1))
	update_val = species_levels[zero_ind]/len(index_vals)
	species_levels[index_vals,:] += update_val
	species_levels = np.delete(species_levels,zero_ind,axis=0)
	cell_points = np.delete(cell_points_with_POI,zero_ind,axis=0)
	if np.shape(cell_points_with_POI)[0] == 1:
		print(cell_points)
		input()
	return cell_points,species_levels
	
	

def single_copy_sim(cell_dtf,A0 = 100,lambda_val = 0.01,alpha = 2,buffer = 5):
	# PLACEHOLDER
	
	cell_coords = cell_dtf[["x_val", "y_val"]].to_numpy().astype(int)
	#display_cell(cell_coords, xrange=[-50, 50], yrange=[-50, 50])
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
	r = 0.01
	epsilon = 56
	tau = 100
	for point_val in points:
		if np.shape(cell_coords)[0] != 0:
			# is the selected point a cell
			indexing_orig = np.where(np.all(cell_coords == point_val, axis=1))
			rand_val = np.random.randint(4)
			# selects location to copy state to
			change_state_loc = np.array((1 - (2 * (rand_val % 2))) * np.array([1 - rand_val // 2, rand_val // 2]))
			change_state_loc += point_val
			indexing_new = np.where(np.all(cell_coords == change_state_loc, axis=1))
			if len(indexing_new[0]) != len(indexing_orig[0]):
				#input()
				
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

def sim_point(cell_dtf,coord,sim_obj,param_dict,grad_drop,grad_base,alt_dose = False):
	# updates a single cells values using a deterministic solution
	A = time.time()
	point_df = cell_dtf.loc[(cell_dtf.x_val==coord[0]) & (cell_dtf.y_val==coord[1])]
	point_df = point_df.rename(columns={"R": "R_0", "B": "B_0", "P": "P_0"})
	dict_val = dict(zip(["R_0", "B_0", "P_0"],point_df[["R_0", "B_0", "P_0"]].to_numpy()[0].tolist()))
	param_dict.update(dict_val)
	param_dict["kprod"] /= cell_dtf.shape[0]
	if coord[0] ==9 and alt_dose:
		grad_base+= grad_drop/2
	param_dict["EGF"] = coord[0]*grad_drop+grad_base

	B = time.time()


	sim_obj.run(param_values=param_dict)

	C = time.time()
	#x = sim.run(n_runs=1, method='ode')
	#y = ScipyOdeSimulator(model,tspan = t_vec, param_values = param_dict)
	D = time.time()
	#results = x.dataframe[["obs_R","obs_B","obs_P"]].iloc[-1]
	cell_dtf.loc[(cell_dtf.x_val == coord[0]) & (cell_dtf.y_val == coord[1]),["R","B","P"]] = sim_obj.y[-1,:]
	E = time.time()
	param_dict["kprod"] *= cell_dtf.shape[0]
	return np.array([B-A,C-B,D-C,E-D])
	#results = results[["R_0", "B_0", 'P_0']].iloc[-1]
	
def sim_points(cell_dtf,sim_obj,param_dict,grad_drop,grad_base,alt_dose = False):
	clocks = np.zeros(4)
	xy_points = cell_dtf[["x_val","y_val"]].to_numpy()
	for point in xy_points:
		clocks += sim_point(cell_dtf,point,sim_obj,cp.copy(param_dict),grad_drop,grad_base,alt_dose = alt_dose)




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
			# Alternative, I think the other is right
			# transition_matrix[point_ind_1] /= np.sum(transition_matrix[point_ind_1] )
			transition_matrix[point_ind_1] /= 1
	#print(np.alltrue(transition_matrix==transition_matrix.T))
	return transition_matrix
	

def diffusion_step(cell_dtf,diffusion_dict):
	# generalize this to all species
	DX = 2
	# max allowed Dx

	connection_matrix = generate_connection_matrix(cell_dtf)
	for species in diffusion_dict.keys():
		Dt = (1 / (4 * diffusion_dict[species]) * DX ** 4)
		Dt = 10
		if Dt > (1 / (4 * diffusion_dict[species]) * DX ** 4):
			print("Timestep too large")
			input()
		diffusion_val = diffusion_dict[species]*Dt/DX**2
		species_diffusion_mat = connection_matrix*diffusion_val
		for point_ind_1 in range(np.shape(species_diffusion_mat)[0]):
			species_diffusion_mat[point_ind_1,point_ind_1] = 1-np.sum(species_diffusion_mat,axis=0)[point_ind_1]
		new_vec = np.matmul(cell_dtf[species].to_numpy(),species_diffusion_mat)
		cell_dtf[species] = new_vec
	
path_to_Chemotaxis = ""
if path_to_Chemotaxis == "":
	try:
		# Obtains the location of the chemotaxis folder if it is in the cwd parents
		path_to_Chemotaxis = Path.cwd().parents[
			[Path.cwd().parents[i].parts[-1] for i in range(len(Path.cwd().parts) - 1)].index(
				"Chemotaxis")]
	except ValueError:
		print("Chemotaxis not found in cwd parents, trying sys.path")
		try:
			# Obtains the location of the Cell_signaling_information folder if it is in sys.path
			path_to_Chemotaxis = Path(sys.path[[Path(i).parts[-1] for i in sys.path].index("Chemotaxis")])
		except ValueError:
			print("Chemotaxis not found in sys.path "
			      "consult 'Errors with setting working directory' in README")
else:
	path_to_Chemotaxis = Path(path_to_Chemotaxis)
os.chdir(path_to_Chemotaxis)

	

# note calculation of area is just np.shape(cell_coords)[0]

R0 = 300000
L = [0,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1]


grad_dist = 1300
units_of_grid = 2
grad_conc = 50
grad_drop = grad_conc/grad_dist*units_of_grid/10
grad_base = 0.03125
print(grad_drop)



# natural surface area of cell
A0 = 100

# maximum surface area of the cell
Amax = 20

# fluctuation energy cost
l = 2

# Adhesion energy
alpha = 15

#display_cell(initial_cell)
diffusion_matrix = {'R':0.025,'B':0.025,'P':0.025}
x_vals = []
y_vals = []
for i in range(10):
	x_vals.append(i)
	y_vals.append(0)

dtf_dict = {'kprod':0.05876103899761891/0.00122,'kbind':0.01625578123538074,'kunbind': 0.39307675090835553,'krp':0.4777926053914779,'krdp':0.04702130222751365,'kdeg':0.000329582245342056,'ksdeg':0.0010486403956587256,'R_0':0,'B_0':0,'P_0':0,'EGF':1}
AZ = time.time()
sims = 0
t_main = np.linspace(0,10,10)
sim_obj = pysb.integrate.Solver(model, tspan=t_main,compiler='cython')
#sim_obj = pysb.simulator.ScipyOdeSimulator(model, tspan=t_main)
distance_val = []
for L_ind in range(len(L)):
	Cell_df = pd.DataFrame({'x_val': x_vals, 'y_val': y_vals, "R": [dtf_dict['kprod']/len(x_vals) for i in range(len(x_vals))],
	                        "B": [0.0 for i in range(len(x_vals))], "P": [0.0 for i in range(len(x_vals))]})
	cnt = 0
	trajectories_9 = np.zeros((400, 3))
	trajectories_0 = np.zeros((400, 3))
	for repeat in range(400):
		
		Z = time.time()
		#Cell_df = single_copy_sim(Cell_df)
		Z = time.time()
		#print(Cell_df)
		A_sum = 0
		B_sum = 0
		for i in range(10):
			cnt+=1
			sim_points(Cell_df,sim_obj,dtf_dict,grad_drop,L[L_ind],alt_dose = (repeat>200))
			diffusion_step(Cell_df,diffusion_matrix)
		sims += Cell_df.shape[0]
		C = time.time()
		print(repeat)
		print(Cell_df[["R","B","P"]].to_numpy())
		trajectories_0[repeat] = Cell_df[["R","B","P"]].to_numpy()[0]
		trajectories_9[repeat] = Cell_df[["R", "B", "P"]].to_numpy()[-1]
	trajectories = np.vstack((trajectories_9[:,-1],trajectories_0[:,-1])).T
	print(trajectories)
	print()
	max_traj = np.max(trajectories,axis=1)
	print(max_traj)
	plt.plot(np.arange(400)*100/60,(trajectories_9[:,-1]-trajectories_0[:,-1]))
	plt.xlabel("time (min)")
	plt.title(f"base: {np.round(np.log10(L[L_ind]),2)}\ngradient: {np.round(np.log10(grad_drop/units_of_grid),2)}")
	plt.ylabel('range')
	plt.savefig(f"figures/L_base_{L_ind}")
	plt.clf()
	distance_val.append(np.max(((trajectories_9[:,-1]-trajectories_0[:,-1])/np.sqrt(max_traj))[200:]))
	#plt.scatter(np.arange(400) * 100 / 60, (trajectories_0[:, -1]),label = "x = 0")
	#plt.scatter(np.arange(400) * 100 / 60, (trajectories_9[:, -1] ),label = "x = 20 um")
	#plt.xlabel("time (min)")
	#plt.title(f"base: {np.round(np.log10(L[L_ind]), 2)}\ngradient: {np.round(np.log10(grad_drop / units_of_grid), 2)}")
	#plt.ylabel('range normalized by sqrt max')
	#plt.show()
plt.plot(np.array(L),distance_val)
plt.xscale('log')
plt.xlabel("Baseline ng/mL")
plt.ylabel("max range (after local excitation)\nnormalized by sqrt max")
plt.show()
