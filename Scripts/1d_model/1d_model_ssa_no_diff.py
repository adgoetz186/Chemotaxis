from Scripts.Models.Simple_Receptor_test_3d import model
import pysb.integrate
import pysb.simulator
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import copy
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
run_dict_no_d_ = {"L0_no_d":{"repeat":10,"L_init":0},"L001_no_d":{"repeat":10,"L_init":0.01},"L01_no_d":{"repeat":10,"L_init":0.1},"L05_no_d":{"repeat":10,"L_init":0.5},"L1_no_d":{"repeat":10,"L_init":1}}
run_dict_d = {"L0_no_d":{"repeat":10,"L_init":0},"L001_no_d":{"repeat":10,"L_init":0.01},"L01_no_d":{"repeat":10,"L_init":0.1},"L05_no_d":{"repeat":10,"L_init":0.5},"L1_no_d":{"repeat":10,"L_init":1}}

#run_dict = {"L05":{"repeat":1,"L_init":0.5}}

run_batch_list = []
for key in run_dict.keys():
    run_batch_list+= [key for i in range(run_dict[key]["repeat"])]

jump_dist = 0.4

grad_dist = 1300
grad_conc = 50
grad_drop = grad_conc/grad_dist/10

for i in range(len(run_batch_list)):
    #convert ligand to position assuming linear gradient
    lmp = run_dict[run_batch_list[i]]['L_init']/grad_drop
    #convert position to ligand level, this is trivial but we will be using both lmp and lmll
    lmll = lmp*grad_drop

    t_init = np.arange(0,86400+60,60)
    t_step = 60
    t_steps = 180
    diffusion_dict = {'diff_R':0.0}
    dtf_dict = {'kprod':0.05876103899761891/0.00122/model.parameters.length.value,'kbind':0.01625578123538074,'kunbind': 0.39307675090835553,'krp':0.4777926053914779,'krdp':0.04702130222751365,'kdeg':0.000329582245342056,'ksdeg':0.0010486403956587256}
    EGF_values = list(np.linspace(lmll,lmll+grad_drop*model.parameters.length.value*model.parameters.dx.value,40))
    #EGF_values = list(np.linspace(0, 1, 40))
    EGF_names = [f"EGF_{k}" for k in range(int(model.parameters.length.value))]
    dict_EGF = dict(zip(EGF_names,EGF_values))
    param_dict = dtf_dict
    param_dict.update(diffusion_dict)
    #param_dict.update(dimensional_dict)
    param_dict.update(dict_EGF)
    start_time = time.time()

    #sim_obj = pysb.integrate.Solver(model, tspan=t_main,param_values=param_dict,compiler='cython')
    sim_obj = pysb.simulator.BngSimulator(model,param_values=copy.copy(param_dict))
    result = sim_obj.run(param_values=copy.copy(param_dict),tspan=t_init,method='ssa')

    print(model.species)
    model_species_string = [str(k) for k in model.species]
    print(model_species_string)
    named_dict = {}
    positionless_species = []
    for j in range(len(model_species_string)):
        #print(i)
        # potentially add code to ensure order is ascending
        val = re.search(r'(_\d+\()',model_species_string[j])
        val = val.group()
        positionless_val = model_species_string[j].replace(val,"(")
        if positionless_val not in named_dict.keys():
            named_dict[positionless_val] = [j]
        else:
            named_dict[positionless_val].append(j)
    print(named_dict)


    print(result.dataframe)
    print(result.dataframe.to_numpy()[-1,:])
    run_dataframe = result.dataframe
    run_dataframe["lmp"] = [lmp for k in range(run_dataframe.shape[0])]
    run_dataframe["lmll"] = [lmll for k in range(run_dataframe.shape[0])]

    print(run_dataframe)

    t = t_init[-1]
    for j in range(t_steps):
        current_state = run_dataframe.to_numpy()[-1,:-2]
        #print(current_state)
        current_state = current_state[np.array(named_dict["EGFR(k='p', ec='b') ** e"])]
        rand_vec = np.random.choice(np.where(current_state >= np.quantile(current_state,.8))[0])
        print(current_state)
        mid = (np.size(current_state)-1) / 2

        lmp += jump_dist*(rand_vec - mid )/ mid
        lmll = lmp*grad_drop
        EGF_values = list(np.linspace(lmll, lmll + grad_drop * model.parameters.length.value * model.parameters.dx.value, 40))
        EGF_names = [f"EGF_{k}" for k in range(int(model.parameters.length.value))]
        dict_EGF = dict(zip(EGF_names, EGF_values))
        param_dict.update(dict_EGF)
        print()
        result = sim_obj.run(param_values=copy.copy(param_dict), tspan=[t,t+t_step], method='ssa', initials=result.dataframe.to_numpy()[-1, :])
        new_run_dataframe = copy.copy(result.dataframe)
        new_run_dataframe["lmp"] = [lmp for k in range(new_run_dataframe.shape[0])]
        new_run_dataframe["lmll"] = [lmll for k in range(new_run_dataframe.shape[0])]
        t+= t_step
        run_dataframe = pd.concat((run_dataframe,new_run_dataframe.iloc[-1:]))
        #print(run_dataframe.to_numpy()[-1])
        #print(model.species)
        #plt.plot(np.arange(np.size(run_dataframe.index.to_numpy())),run_dataframe.index.to_numpy())
        #plt.show()
    # it looks like there might be some type of rounding error in the assignment of initial values
    #result_2 = sim_obj.run(param_values=param_dict,tspan=t_main,method='ode',initials = result.dataframe.to_numpy()[-1,:])
    #print(result_2.dataframe)
    #print(result_2.dataframe.to_numpy()[-1, :])
    #input()
    if not os.path.exists(f"data/Simulations_1d_no_diff_stoch/{run_batch_list[i]}"):
        os.mkdir(f"data/Simulations_1d_no_diff_stoch/{run_batch_list[i]}")
    run_dataframe.to_csv(f"data/Simulations_1d_no_diff_stoch/{run_batch_list[i]}/{run_batch_list[i]}_{len(os.listdir(f'data/Simulations_1d_no_diff_stoch/{run_batch_list[i]}'))}")
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
