import numpy as np
import copy as cp
import cython
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

import pysb.integrate

from Scripts.Models.diffusion_test import model
from pysb.simulator import BngSimulator
# import tensorflow as tf
import pandas as pd





path_to_CSI = ""
if path_to_CSI == "":
	try:
		# Obtains the location of the chemotaxis folder if it is in the cwd parents
		path_to_CSI = Path.cwd().parents[
			[Path.cwd().parents[i].parts[-1] for i in range(len(Path.cwd().parts) - 1)].index(
				"chemotaxis")]
	except ValueError:
		print("Cell_signalling_information not found in cwd parents, trying sys.path")
		try:
			# Obtains the location of the Cell_signaling_information folder if it is in sys.path
			path_to_CSI = Path(sys.path[[Path(i).parts[-1] for i in sys.path].index("chemotaxis")])
		except ValueError:
			print("chemotaxis not found in sys.path "
			      "consult 'Errors with setting working directory' in README")
else:
	path_to_CSI = Path(path_to_CSI)
os.chdir(path_to_CSI)

# note calculation of area is just np.shape(cell_coords)[0]

R0 = 300000
L = [0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 100]

grad_dist = 1300
units_of_grid = 2
grad_conc = 50
grad_drop = grad_conc / grad_dist * units_of_grid / 10
grad_base = 0.03125
print(grad_drop)
dtf_dict = {'kprod': 0.05876103899761891 / 0.00122, 'diff': 0.01625578123538074}

A = time.time()
#point_df = cell_dtf.loc[(cell_dtf.x_val==coord[0]) & (cell_dtf.y_val==coord[1])]
#point_df = point_df.rename(columns={"R": "R_0", "B": "B_0", "P": "P_0"})
#dict_val = dict(zip(["R_0", "B_0", "P_0"],point_df[["R_0", "B_0", "P_0"]].to_numpy()[0].tolist()))
#param_dict.update(dict_val)
#param_dict["kprod"] /= cell_dtf.shape[0]
#param_dict["EGF"] = coord[0]*grad_drop+grad_base
#param_dict["EGF"] = 0
#print(param_dict["kprod"],cell_dtf.shape[0], 1)
#param_dict.update(point_df[["R_0","B_0","P_0"]])
B = time.time()
#print(param_dict.values())
t_main = np.linspace(0,100,10)
sim_obj = pysb.integrate.Solver(model, tspan=t_main)
sim_obj.run(param_values=dtf_dict)
print(sim_obj)
input()

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

# print(Cell_df.shape[0])
# print(np.sum(Cell_df[["R"]].to_numpy()))
# print(np.average(Cell_df[["x_val","y_val"]].to_numpy(),axis=0))

# print((A-Z)/(C-Z),(B-A)/(C-Z),(C-B)/(C-Z))
# print((-1*(AZ-time.time())/(sims)))

# print(dtf_dict)
