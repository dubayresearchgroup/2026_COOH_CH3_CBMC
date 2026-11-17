import os
import sys

# sys.path.insert(0,os.path.abspath('/mnt/c/AGSAMproj/mynpmc/np-mc')) #Necessary to pull modules from np-mc
import npmc.move_class
import npmc.simulation_class as simc

# import npmc.eval_functions as ev

# jobID = sys.argv[1]
# ev.pushStart(os.getcwd(),'mpa_ddt.slurm',jobID)
init_file = os.path.abspath("./system.in")
datafile = os.path.abspath("./system.data")
dumpfile = os.path.abspath("./trajectory.xyz")
temp = 298.15

sim = simc.Simulation(
    init_file,
    datafile,
    dumpfile,
    temp,
    anchortype=5,
    max_disp=0.4,
    jump_dists=[0.93, 1.95],
    numtrials_jump=20,
    type_lengths=(13, 7),
    moves=[1, 10, 1, 10, 1],
    read_pdf=True,
)  # seed=int(jobID) this goes after the moves variable in sim = simc.simulation.....
sim.exclude_type(1, 1)
sim.minimize(max_iter=500)
sim.initial_PE = sim.get_total_PE()

numsteps = 4000000
for i in range(numsteps):
    if (i % 10000) == 0:
        sim.dump_atoms()
    sim.perform_mc_move(temp=temp)
