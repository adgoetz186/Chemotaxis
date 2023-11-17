from Scripts.Models.Simple_Receptor_test_3d import model
import pysb.integrate
import pysb.simulator
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import os
import sys


path_to_CSI = ""
if path_to_CSI == "":
	try:
		# Obtains the location of the chemotaxis folder if it is in the cwd parents
		path_to_CSI = Path.cwd().parents[
			[Path.cwd().parents[i].parts[-1] for i in range(len(Path.cwd().parts) - 1)].index(
				"Chemotaxis")]
	except ValueError:
		print("Chemotaxis not found in cwd parents, trying sys.path")
		try:
			# Obtains the location of the Cell_signaling_information folder if it is in sys.path
			path_to_CSI = Path(sys.path[[Path(i).parts[-1] for i in sys.path].index("Chemotaxis")])
		except ValueError:
			print("Chemotaxis not found in sys.path "
			      "consult 'Errors with setting working directory' in README")
else:
	path_to_CSI = Path(path_to_CSI)
os.chdir(path_to_CSI)


import pandas as pd
L = [0,0.0078125,0.015625,0.03125,0.0625,0.125,0.25,0.5,1]
run_dict = {"L0":{"repeat":10,"L_init":0},"L128":{"repeat":10,"L_init":0.0078125},"L64":{"repeat":10,"L_init":0.015625},"L32":{"repeat":10,"L_init":0.03125},"L16":{"repeat":10,"L_init":0.0625},"L8":{"repeat":10,"L_init":0.125},"L4":{"repeat":10,"L_init":0.25},"L2":{"repeat":10,"L_init":0.5},"L1":{"repeat":10,"L_init":1.0}}
run_batch_list = []
for key in run_dict.keys():
    run_batch_list+= [key for i in range(run_dict[key]["repeat"])]
print(run_batch_list)


grad_dist = 1300
grad_conc = 50
grad_drop = grad_conc/grad_dist/10*model.parameters.length.value*model.parameters.dx.value

for i in range(len(run_batch_list)):
    lmll = run_dict[run_batch_list[i]]
    t_main = np.linspace(0,50000,1000)
    diffusion_dict = {'diff_R':0.025}
    #diffusion_dict = {'diff_R': 0.000025}
    #diffusion_dict = {'diff_R': 0.25}
    #diffusion_dict = {'diff_R': 250}
    #dimensional_dict = {'dx':0.1,'length':5}
    dtf_dict = {'kprod':0.05876103899761891/0.00122/model.parameters.length.value,'kbind':0.01625578123538074,'kunbind': 0.39307675090835553,'krp':0.4777926053914779,'krdp':0.04702130222751365,'kdeg':0.000329582245342056,'ksdeg':0.0010486403956587256,'R_0':0,'B_0':0,'P_0':0}
    EGF_values = list(np.linspace(run_dict[run_batch_list[i]]["L_init"],run_dict[run_batch_list[i]]["L_init"]+grad_drop,40))
    #EGF_values = list(np.linspace(0, 1, 40))
    EGF_names = [f"EGF_{i}" for i in range(int(model.parameters.length.value))]
    dict_EGF = dict(zip(EGF_names,EGF_values))
    param_dict = dtf_dict
    param_dict.update(diffusion_dict)
    #param_dict.update(dimensional_dict)
    param_dict.update(dict_EGF)
    start_time = time.time()
    #sim_obj = pysb.integrate.Solver(model, tspan=t_main,param_values=param_dict,compiler='cython')
    sim_obj = pysb.simulator.BngSimulator(model, tspan=t_main,param_values=param_dict)
    result = sim_obj.run(param_values=param_dict,method='ssa')
    if not os.path.exists(f"data/Simulations_1d_diff_stoch/{run_batch_list[i]}"):
        os.mkdir(f"data/Simulations_1d_diff_stoch/{run_batch_list[i]}")
    result.dataframe.to_csv(f"data/Simulations_1d_diff_stoch/{run_batch_list[i]}/{run_batch_list[i]}_{i}")
    print(result.dataframe)
    print(time.time()-start_time, i)








    #input()






    #print(model.species)
    #print(sim_obj.y)
    #list_of_locs = []
    #list_of_species = []

    #for i in model.species:
        # print(i)
    #    if int(str(i).split("_")[1].split("(")[0]) == 0:
    #        print(str(i))
    #        val = str(i).split("_")[0] + "(" + str(i).split("_")[1].split("(")[1]
    #        if val not in list_of_species:
    #            list_of_species.append(val)
    #print(list_of_species)
    #print(model.species)
    #mdl_species = np.array([str(i) for i in model.species])
    #dict_of_ind = {}
    #for species in list_of_species:
    #    species_ind_list = []
    #    for i in range(len(model.species)):
    #        if species == str(model.species[i]).split("_")[0] + "(" + str(model.species[i]).split("_")[1].split("(")[1]:
    #            species_ind_list.append(i)
    #    print(species_ind_list)
    #    dict_of_ind[species] = np.array(species_ind_list)
    #    print(dict_of_ind)
    #    print(mdl_species[species_ind_list])
    #    print(result.dataframe.to_numpy()[:,species_ind_list])
    #    input()
