import numpy as np
from ase.io import read,Trajectory
from mace.calculators import MACECalculator
from ase.build.surface import mx2
from ase.visualize import view
import os
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

calc = MACECalculator(model_paths="../training/bilayer_pes.model",device="cuda", default_dtype="float64")

# 1-> XX' 2-> 2 3-> MX' 4-> 4 5 -> XM' 6 -> f(d)  
P_offsets = np.array([[0,0],[0,1/(2*np.sqrt(3))],
            [0,-1/np.sqrt(3)],[0,1*np.sqrt(3)/2],
            [0,1/np.sqrt(3)],[-1/3,1/np.sqrt(3)]])

# 1-> XX' 2-> 2 3-> 2H 4-> 4 5 -> MM' 6 -> f(d) 
AP_offsets = np.array([[0,0],[0,-1/(2*np.sqrt(3))],
            [0,-1/np.sqrt(3)],[0,1/(2*np.sqrt(3))],
            [0,1/np.sqrt(3)],[-1/3,0]])

def get_tmdc_bilayer(tmdc1,tmdc2,N,offset,d,parallel):

    # Lattice constants
    a = {'WS2': 3.190902132, 'MoS2': 3.186305534, 'WSe2': 3.322874304, 'MoSe2': 3.322339468}
    t = {'WS2': 3.1570497014, 'MoS2': 3.140170013, 'WSe2': 3.367178811, 'MoSe2': 3.3492559072}

    ## Make tmdc monolayers and align them with the axes
    sup_tmdc1 = mx2(tmdc1,kind='2H',a=a[tmdc1],thickness=t[tmdc1],vacuum=8,size=(N,N,1))
    sup_tmdc1.rotate(60,[0,0,1],rotate_cell=True)
    sup_tmdc2 = mx2(tmdc2,kind='2H',a=a[tmdc2],thickness=t[tmdc2],vacuum=8,size=(N,N,1))
    sup_tmdc2.rotate(60,[0,0,1],rotate_cell=True)

    #Rotate 180 degrees to get XX' antiparallel stacking
    if not parallel:

        sup_tmdc2.rotate(180,sup_tmdc2.positions[1]-sup_tmdc2.positions[2],
                        center=(sup_tmdc2.positions[1]+sup_tmdc2.positions[2])/2)

    #Get unit lattice vectors
    unit_v1 = np.array([1,0,0])
    unit_v2 = np.array([0,1,0])
    unit_v3 = np.array([0,0,1])

    #Get translation vector from offset and interlayer distance d
    translate_v1 = a[tmdc2]*offset[0]*unit_v1
    translate_v2 = a[tmdc2]*offset[1]*unit_v2
    translate_v3  = (d)*unit_v3

    sup_tmdc2.translate(translate_v1+translate_v2+translate_v3)

    bilayer = sup_tmdc1 + sup_tmdc2

    return bilayer


timesteps = 10000
temp = 300
femto_dt = 1 
dt = femto_dt*units.fs
N = 6
foldername = 'aligned_bilayer_md_data'
offset = P_offsets[4]
nruns = 50

for m1,m2 in [('Mo','Mo'),('W','W'),('Mo','W')]:
#for m1,m2 in [('W','W')]:
    for ch in ['S','Se']:
        
        if ch=='Se':
            interlayer_d = 6.5
        else:
            interlayer_d = 6.2
            
        for parallel in [True]:
            
            if parallel == True:
                stacking = '3R2'
            else:
                stacking = '2H'
                
            print(f'{stacking}-{m1}{ch}2/{m2}{ch}2')

            for run_no in range(nruns):
                
                print(f'Run {run_no}')
                
                trajname = f'{m1}{m2}{ch}_{stacking}_T{temp:04.0f}_{femto_dt}fs_i{timesteps}_run{run_no:02}'
                trajfile = f'{foldername}/{trajname}.traj'
                
                if not os.path.exists(trajfile) or os.path.getsize(trajfile) == 0:
                
                    tout = Trajectory(trajfile,"w")
                    atoms = get_tmdc_bilayer(f'{m1}{ch}2',f'{m2}{ch}2',N,offset,interlayer_d,parallel)
                    atoms_md = atoms.copy()
                    atoms_md.calc = calc
                    MaxwellBoltzmannDistribution(atoms_md, temperature_K=temp)
                    Stationary(atoms_md)
                    dyn = Langevin(atoms_md, timestep=dt, temperature_K=temp, friction=0.01,
                         trajectory=None, logfile=None)
                    dyn.run(1000)
                    t = 0
                    
                else:
                    tout = Trajectory(trajfile,"a")
                    atoms_md = read(trajfile, index=-1) 
                    atoms_md.calc = calc
                    dyn = Langevin(atoms_md, timestep=dt, temperature_K=temp, friction=0.01,
                         trajectory=None, logfile=None)
                    t = len(tout)
                    
                while t<timesteps:
                    tout.write(atoms_md)
                    dyn.run(1)
                    t+=1
                tout.close()
