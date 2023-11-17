from Scripts.Models.Simple_Receptor_test_3d import model
import pysb.integrate
import pysb.simulator
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import pandas as pd
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


L1_1 = pd.read_csv("data/Simulations_1d_diff_stoch/L001/L001_0").to_numpy()
print(np.sum(L1_1[:,1:],axis = 1))
plt.plot(L1_1[:,0],np.sum(L1_1[:,1:],axis = 1))
plt.show()