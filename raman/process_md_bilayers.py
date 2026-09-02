from mace.calculators import MACECalculator
import numpy as np
import os
from ase import units
from ase.io import read, Trajectory
import sys

calc_raman  = MACECalculator(model_paths="../training/aligned_bilayer_dielectrics.model",
                             model_type="DipoleMACE",device="cuda", default_dtype="float64")

m1 = sys.argv[1]
m2 = sys.argv[2]
ch = sys.argv[3]
stacking = sys.argv[4]
femto_dt = int(sys.argv[5])
temp = float(sys.argv[6])
timesteps = int(sys.argv[7])
run_no = int(sys.argv[8])

dt = femto_dt*units.fs
foldername = 'aligned_bilayer_md_data'

suscs = np.zeros((timesteps,3,3))
times = np.zeros(timesteps)
trajname = f'{m1}{m2}{ch}_{stacking}_T{temp:04.0f}_{femto_dt}fs_i{timesteps}_run{run_no:02}'
trajfile = f'{foldername}/{trajname}.traj'

traj = Trajectory(f'{foldername}/{trajname}.traj')

for t,atoms_md in enumerate(traj):
    calc_raman.calculate(atoms_md)
    suscs[t]  = calc_raman.results['polarizability']
    times[t] = t*dt
np.savez(f'{foldername}/{trajname}.npz',times=times,suscs=suscs)
