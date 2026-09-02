from mace.calculators import MACECalculator
import numpy as np
from ase import units
from ase.io import read, Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import sys
from utils import *

calc = MACECalculator(model_paths="../training/alloys_pes.model", device="cuda", default_dtype="float64")
calc_raman  = MACECalculator(model_paths="../training/alloys_dielectric.model",
                             model_type="DipoleMACE",device="cuda", default_dtype="float64")

x = float(sys.argv[1])
y = float(sys.argv[2])
N = int(sys.argv[3])         
femto_dt = int(sys.argv[4])
temp = float(sys.argv[5])
timesteps = int(sys.argv[6]) 
run_no = int(sys.argv[7])

dt = femto_dt*units.fs
foldername = 'monolayer_md_data'

atoms = make_alloy(x,y,N)

atoms_md = atoms.copy()
atoms_md.calc = calc
MaxwellBoltzmannDistribution(atoms_md, temperature_K=temp)
Stationary(atoms_md)
dyn = Langevin(atoms_md, timestep=dt, temperature_K=temp, friction=0.01,
     trajectory=None, logfile=None)

## MD Equilibration
dyn.run(1000)
suscs = np.zeros((timesteps,3,3))
times = np.zeros(timesteps)
trajname = f'x{x:.3f}y{y:.3f}T{temp:04.0f}_{femto_dt}fs_i{timesteps}_run{run_no:02}'
tout = Trajectory(f'{foldername}/{trajname}.traj',"w")

for t in range(timesteps):
    tout.write(atoms_md)
    calc_raman.calculate(atoms_md)
    suscs[t]  = calc_raman.results['polarizability']
    times[t] = t*dt
    dyn.run(1)
    if t%1000==0:
        print(f' {temp} {t}  {suscs[t]}')
tout.close()
np.savez(f'{foldername}/{trajname}.npz',times=times,suscs=suscs)
