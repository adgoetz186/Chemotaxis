from pysb import *

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

# Due to the replacement rate of the ligand, the ligand concentration acts like a constant
# so IGF binding acts as a change of state of IGFR rather than a binding event
Monomer('R', ['locx'])

Parameter('kprod',1.0)
#print(kprod.value)
# this is effective binding constant, should be igf concentration and rate constant
Parameter('diff',1.0)



Rule('r_prod', None >> R(locx=0)**e,kprod)
Rule('diff', R(locx=0)**e >> R(locx=1)**e,diff)


Observable('obs_R', EGFR())
Observable('obs_R2', EGFR(locx=1))

