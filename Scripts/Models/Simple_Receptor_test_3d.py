from pysb import *
import numpy as np

# Parameters in order:
# kprod - unbound unphosphorylated receptor synthesis rate
# kbind - receptor ligand binding rate constant
# kunbind - receptor ligand unbinding rate constant
# kp - receptor phosphorylation rate constant
# kdp - receptor dephosphorylation rate constant
# kdeg - non activated receptor degredation rate
# ksdeg - activated receptor degredation rate
# EGFR_0 - Initial unphosphorylated unbound EGFR count


Model()
Compartment('e',dimension=2,parent=None)
Parameter('dx', 0.5)
Parameter('length', 40)


Parameter('kprod',1.0)
#print(kprod.value)
# this is effective binding constant, should be igf concentration and rate constant
Parameter('kbind',1.0)
Parameter('kunbind',1.0)
Parameter('krp',1.0)
Parameter('krdp',1.0)
Parameter('kdeg',1.0)
Parameter('ksdeg',1.0)
Parameter('diff_R', 1.0)


for i in range(int(length.value)):
    Parameter(f'EGF_{i}', 1.0)
    eval(f"Expression(f'keffbind_{i}', kbind * EGF_{i})")

# Due to the replacement rate of the ligand, the ligand concentration acts like a constant
# so IGF binding acts as a change of state of IGFR rather than a binding event
for i in range(int(length.value)):
    Monomer(f'EGFR_{i}', ['k','ec'],{'k':['up','p'],'ec':['b','ub']})

#for i in range(int(length.value)):
#    Parameter(f'R_0_{i}', 0)
#    Parameter(f'B_0_{i}', 0)
#    Parameter(f'P_0_{i}', 0)
#    eval(f"Initial(EGFR_{i}(k='up',ec='ub')**e, f'R_0_{i}')")
#    eval(f"Initial(EGFR_{i}(k='up',ec='b')**e, f'B_0_{i}')")
#    eval(f"Initial(EGFR_{i}(k='p',ec='b')**e, f'P_0_{i}')")

connection_matrix = np.zeros((int(length.value),int(length.value)))
for i in range(np.shape(connection_matrix)[0]):
    if i == 0 and i < (np.shape(connection_matrix)[0]-1):
        connection_matrix[i, i + 1] = 1
    elif i == (np.shape(connection_matrix)[0]-1):
        connection_matrix[i, i - 1] = 1
    else:
        connection_matrix[i,i+1] = 1
        connection_matrix[i, i-1] = 1
    #connection_matrix[i] /= np.sum(connection_matrix[i])
print(connection_matrix)

for i in range(np.shape(connection_matrix)[0]):
    for j in range(np.shape(connection_matrix)[1]):
        if connection_matrix[i,j] != 0:
            Expression(f"kdiff_{i}_{j}", connection_matrix[i,j]*diff_R/dx**2)
            eval(f"Rule('r_diff_{i}_{j}', EGFR_{i}(k='up',ec='ub')**e >> EGFR_{j}(k='up',ec='ub')**e,kdiff_{i}_{j})")
            eval(f"Rule('b_diff_{i}_{j}', EGFR_{i}(k='up',ec='b')**e >> EGFR_{j}(k='up',ec='b')**e,kdiff_{i}_{j})")
            eval(f"Rule('p_diff_{i}_{j}', EGFR_{i}(k='p',ec='b')**e >> EGFR_{j}(k='p',ec='b')**e,kdiff_{i}_{j})")

for i in range(int(length.value)):
    eval(f"Rule('r_prod_{i}', None >> EGFR_{i}(k='up',ec='ub')**e,kprod)")
    eval(f"Rule('r_deg_{i}', EGFR_{i}(k='up')**e >> None ,kdeg)")
    eval(f"Rule('rp_deg_{i}', EGFR_{i}(k='p')**e >> None ,ksdeg)")
    eval(f"Rule('r_bind_{i}', EGFR_{i}(ec='ub',k='up')**e | EGFR_{i}(ec='b',k='up')**e,keffbind_{i},kunbind)")
    eval(f"Rule('rb_phos_{i}', EGFR_{i}(k='up',ec='b')**e | EGFR_{i}(k='p',ec='b')**e,krp,krdp)")


