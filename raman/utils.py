from ase.build.surface import mx2
import numpy as np
import os
import matplotlib.pyplot as plt
from ase import units
from ase.io import read, Trajectory
import sys
import numpy as np
from scipy.fftpack import fft
from scipy.signal import correlate
import matplotlib.pyplot as plt
import math
from typing import Optional
import pandas as pd
from matplotlib.ticker import MultipleLocator
import scipy.signal as ssl
from brokenaxes import brokenaxes
import matplotlib.lines as mlines
from scipy import fftpack
from ase.vibrations import Vibrations
from ase.vibrations import VibrationsData
from scipy.interpolate import LinearNDInterpolator
from ase.optimize import BFGS
from ase.build.surface import mx2
from ase.filters import FrechetCellFilter

a_optB88_QE = {
    (1, 0): 3.190902132,
    (0, 0): 3.186305534,
    (1, 1): 3.322874304,
    (0, 1): 3.322339468
}

t_optB88_QE = {
    (1, 0): 3.1570497014,
    (0, 0): 3.140170013,
    (1, 1): 3.367178811,
    (0, 1): 3.3492559072
}

# Prepare data for interpolation
xd = np.array([0, 1, 0, 1])
yd = np.array([0, 0, 1, 1])
ad = np.array([a_optB88_QE[(x, y)] for x, y in zip(xd, yd)])
td = np.array([t_optB88_QE[(x, y)] for x, y in zip(xd, yd)])

# Create LinearNDInterpolator instances
a_xy = LinearNDInterpolator(list(zip(xd, yd)), ad)
t_xy = LinearNDInterpolator(list(zip(xd, yd)), td)

def make_alloy(x,y,N):
    '''
    Make MoWSSe alloy having formula Mo(1-x)W(x)S(2-2y)Se(2y)
    '''
    prim_cell = mx2(formula='MoS2',kind='2H',a=a_xy(x,y),thickness=t_xy(x,y),vacuum=8,size=(1,1,1))
    atoms = prim_cell*(N,N,1)
    nflip_x = int(len(atoms)/3*x)
    flip_x = np.random.choice(range(0,len(atoms),3),nflip_x,replace=False)
    for f in flip_x:
        atoms.symbols[f] = 'W'

    #Flip S with Se
    nflip_y = int(2*len(atoms)/3*y)
    flip_y = np.random.choice([i for i in range(len(atoms)) if i%3!=0],nflip_y,
                              replace=False)
    for f in flip_y:
        atoms.symbols[f] = 'Se'
    return atoms

def get_eigenvectors_and_freq(atoms,calc):
    '''
    Get eigenvectors for alloy supercell 
    '''
    
    atoms.calc = calc
    dyn = BFGS(atoms,logfile=None)
    #dyn.log = passlog
    dyn.run(fmax=0.001)
    vib = Vibrations(atoms)
    vib.clean()
    vib.run()
    vib_data = vib.get_vibrations()
    energies,sc_eige = vib_data.get_energies_and_modes()
    vib.clean()
    frequencies = energies/ase.units.invcm
    
    return sc_eige,frequencies

def get_effective_raman(x,y):
    '''
    Get effective Raman tensor for (x,y) composition
    '''
    raman_mos2 = read_raman('ph_data/mos2.ph.rec.out')
    raman_mose2 = read_raman('ph_data/mose2.ph.rec.out')
    raman_ws2 = read_raman('ph_data/ws2.ph.rec.out')
    raman_wse2 = read_raman('ph_data/wse2.ph.rec.out')
    
    effective_tensor = (1-x)*(1-y)*raman_mos2 + x*(1-y)*raman_ws2 + (1-x)*y*raman_mose2 + x*y*raman_wse2
    
    return effective_tensor 

def get_sc_tensors_effective(sup_eige,R_pc):
    '''
    Get raman tensors for alloy supercell
    using effective raman tensors obtained from supercells
    '''
    raman_tensors = np.zeros((sup_eige.shape[0],3,3))
    natp = int(R_pc.shape[0]/3)
    
    for nus in range(sup_eige.shape[0]):
        for iats in range(sup_eige.shape[1]):
            for lp in range(3):
                raman_tensors[nus,:,:] += sup_eige[nus,iats,lp] * R_pc[(iats % natp)*3+lp,:,:]
    
    return raman_tensors

def get_sc_tensors(sup_eige,atoms,calc_raman):

    calc_raman.calculate(atoms)
    dchi_dtau = calc_raman.results['polarizability_deriv']
    dchi_dtau = -dchi_dtau.reshape(len(atoms),3,3,3).transpose((0, 3, 1, 2)).reshape(len(atoms)*3,3,3)
    raman_tensors = np.zeros((sup_eige.shape[0],3,3))      
    for nus in range(sup_eige.shape[0]):
        for iats in range(sup_eige.shape[1]):
            for lp in range(3):
                raman_tensors[nus,:,:] += sup_eige[nus,iats,lp]*dchi_dtau[iats*3+lp,:,:]

    return raman_tensors

def get_raman_intensities(raman_tensors):
    '''
    Get Raman intensities for the tensors
    '''
    raman_intensities = np.zeros(raman_tensors.shape[0])
    
    for i in range(raman_tensors.shape[0]):
        
        tensor = raman_tensors[i]
        r_xx,r_yy,r_zz  = tensor.diagonal() ##diagonal terms
        r_xy,r_yz,r_xz = tensor[[0,1,2],[1,2,0]] ##off diagonal terms
        
        #Use formula 45a^2 + 7gamma^2
        a = (r_xx+r_yy+r_zz)/3
        gamma_sq = 0.5*((r_xx-r_yy)**2+(r_xx-r_zz)**2+(r_yy-r_zz)**2
                        +6*(r_xy**2 + r_yz**2 + r_xz**2))
        
        intensity = 45*a**2 + 7*gamma_sq
        
        raman_intensities[i] = intensity
        
    return raman_intensities  

def read_dft_spectra(filename,nmodes):
    '''
    Read frequencies and intensities from dynmat output file
    '''
    file = open(filename)
    lines = file.readlines()

    freqs = np.zeros(nmodes)
    intens = np.zeros(nmodes)
    
    for k,line in enumerate(lines):
        if '# mode' in line:
            k=k+1
            for i in range(nmodes):

                freqs[i],intens[i] = float(lines[k].split()[1]),float(lines[k].split()[4])
                k+=1

            break
    return freqs,intens


def gaussian(x,amplitude,mean,std):
    '''
    Gaussian curve A*e^(-(x-mean)^2/2*std^2) 
    '''
    return amplitude*np.exp(-(x-mean)**2/(2*std**2))

def get_spectra_curve(x,intensities,frequencies,std):
    '''
    Get Raman spectrum by summing up gaussians centered at 
    vibrational frequencies,and peak amplitude as the 
    corresponding intensities
    '''
    y = np.zeros(len(x))

    for i,intensity in enumerate(intensities):
        
        if not np.isreal(frequencies[i]):
            continue
            
        y = y + gaussian(x,intensity,frequencies[i],std)
    return x,y

def rgb_to_cmyk(r, g, b):
    # Convert RGB values to range of 0-1
    r, g, b = r/255.0, g/255.0, b/255.0

    # Find the maximum value of RGB values
    max_value = max(r, g, b)

    # If max_value is 0, return 0, 0, 0, 1
    if max_value == 0:
        return 0, 0, 0, 1

    # Calculate the K value
    k = 1 - max_value

    # Calculate the C, M, and Y values
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    # Return the CMYK values
    return c, m, y, k

def cmyk_to_rgb(c, m, y, k):
    # Calculate the RGB values
    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    # Round the RGB values and return them as integers
    return int(round(r)), int(round(g)), int(round(b))

def get_alloy_color(x,y):
    
    #datapoints = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] #1-x,x,2-2y,2y
    
    valuepoints = np.array([[0.8, 0.8, 0.0, 0.2],  #Mo
                            [0.0, 0.0, 0.8, 0.2],  #W
                            [0.0, 0.8, 0.0, 0.2],  #S
                            [0.8, 0.0, 0.0, 0.2]]) #Se     in percentage
    
    cmyk_color = (x*valuepoints[0] + (1-x)*valuepoints[1] + y*valuepoints[2] + (1-y)*valuepoints[3])/2
    
    return tuple(np.array(cmyk_to_rgb(*cmyk_color))/255)

def get_max_peak(xdata,ydata,xmin,xmax,return_index=False):
    
    idx = np.where((xdata>xmin) & (xdata<xmax))[0]
    peaks = ssl.find_peaks(ydata[idx])[0]
    #print(peaks[0])
    biggest_peak_idx = peaks[np.argmax(ydata[idx][peaks])]
    global_idx = idx[biggest_peak_idx]
    biggest_peak_xdata = xdata[global_idx]
    biggest_peak_ydata = ydata[global_idx]
    
    if return_index:
        return biggest_peak_xdata, biggest_peak_ydata, global_idx
    else:
        return biggest_peak_xdata, biggest_peak_ydata

def get_fwhm(x, y,ref_y,peak_index):
    """Calculate FWHM of a peak given x, y arrays and peak index."""
    if peak_index is None:
        return np.nan
    half_max = (y[peak_index]+ref_y)/ 2
    left_idx = peak_index
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    right_idx = peak_index
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    if left_idx == 0 or right_idx == len(y) - 1:
        return np.nan  # Cannot determine width properly
    x1 = np.interp(half_max, [y[left_idx], y[left_idx + 1]], [x[left_idx], x[left_idx + 1]])
    x2 = np.interp(half_max, [y[right_idx - 1], y[right_idx]], [x[right_idx - 1], x[right_idx]])
    return x2 - x1  
  

def polarisability_derivative(pol, dt):
    '''
    Compute time derivative of polarisability tensor using central differences.
    '''
    prev_pol = pol[:-2]
    next_pol = pol[2:]
    dpol_dt = (next_pol - prev_pol) / (2 * dt)

    return dpol_dt

def autocorr(time_series):
    '''
    Calculate time autocorrelation
    '''
    corr = correlate(time_series,time_series)
    return corr

def windowFunction(trajLength):
    # calculate window function in form of the Hann function to minimize the effect of the finite length of the trajectory
    winFunc = np.arange(0, trajLength)*math.pi/(2*(trajLength-1))
    winFunc = np.cos(winFunc)
    return winFunc

def fourierTransform(signal,dt):
    winFunc = windowFunction(signal.shape[0])
    # perform transformation by window function
    signal = signal * winFunc
    spectrum = math.sqrt(2*np.pi)*np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    return spectrum,freqs

def get_a_pol(pol,dt):
    iso_autocorr = autocorr(pol[:,0,0]+pol[:,1,1]+pol[:,2,2])/9
    return fourierTransform(iso_autocorr,dt)


def get_gamma_pol(pol,dt):
    # calculate derivative of polarizability
    # calculate average time correlation function of derivative of polarizability
    aniso_autocorr = autocorr(pol[:,0,0]-pol[:,1,1])/2
    aniso_autocorr = aniso_autocorr + autocorr(pol[:,1,1]-pol[:,2,2])/2
    aniso_autocorr = aniso_autocorr + autocorr(pol[:,2,2]-pol[:,0,0])/2
    aniso_autocorr = aniso_autocorr + 3*autocorr(pol[:,0,1])
    aniso_autocorr = aniso_autocorr + 3*autocorr(pol[:,1,2])
    aniso_autocorr = aniso_autocorr + 3*autocorr(pol[:,2,0])
    # perform fourier transform
    return fourierTransform(aniso_autocorr, dt)

def get_prefactor(freq_spectrum,T,lambdaLaser=532e-9,reduced=False):

    # Planck constant h [Js]
    h = 6.62607015e-34
    hQuer = h/(2*math.pi)

    # Boltzmann constant kB [J/K]
    kB = 1.380649e-23
    # Vacuum permittivity epsilon0 [C/Vm]
    epsilon = 8.8541878128e-12
    # Speed of light [m/s]
    c = 299792458
    #wIn = lambdaLaser/c

     
    wIn = 2 * np.pi * c / lambdaLaser
    preF = hQuer/(8*pow(math.pi, 3)*pow(epsilon, 2)*pow(c, 4)*kB*T)
    
    if not reduced:
        preF *= pow((wIn - freq_spectrum), 4)    
        preF /= freq_spectrum*(1-np.exp(-hQuer*freq_spectrum/(kB*T)))

    return preF

def get_contributions(pol,dt):
    dpol_dt = polarisability_derivative(pol,dt)
    a,freqs = get_a_pol(dpol_dt,dt)
    gamma,freqs = get_gamma_pol(dpol_dt,dt)
    return a[1:], gamma[1:], freqs[1:]


def calculate_average_spectra(foldername, x, y, temp, timesteps, nsteps, nruns,
                              interval, xspace, freq_lim, std, x_layer2=None,
                              derivative=False, polarisation='unpolarised',reduced=False):
    
    total_runs = 0
    
    x_el = {0: 'Mo', 1: 'W'}
    y_el = {0: 'S', 1: 'Se'}
    
    for run_no in range(nruns):
        
        if x_layer2:
            trajname = f'{x_el[x]}{x_el[x_layer2[0]]}{y_el[y]}_{x_layer2[1]}_'\
                       f'T{temp:04.0f}_1fs_i{timesteps}_run{run_no:02}'
        else:
            trajname = f'x{x:.3f}y{y:.3f}T{temp:04.0f}_1fs_i{timesteps}_run{run_no:02}'
        
        npzfile = f'{foldername}/{trajname}.npz'
        
        if not os.path.exists(npzfile):
            continue
        
        total_runs += 1
        data = np.load(npzfile)
        times = data['times'][:nsteps:interval]
        suscs = data['suscs'][:nsteps:interval]
        dt = 1e-15 * (times[1] - times[0]) / units.fs
        
        a, gamma, freqs = get_contributions(suscs, dt)
        
        if polarisation == 'unpolarised':
            intens = np.abs(45 * a + 7 * gamma) / 45
        elif polarisation == 'cross':
            intens = np.abs(3 * gamma) / 45
        elif polarisation == 'parallel':
            intens = np.abs(45 * a + 4 * gamma) / 45
        
        preF = get_prefactor(freqs, temp, lambdaLaser=532e-9,reduced=reduced)
        intens = preF * intens
        
        c = 299792458
        freqs_cm = freqs / (c * 1e2)
        
        # Filter using low and high limits
        low_lim, high_lim = freq_lim
        mask = (freqs_cm >= low_lim) & (freqs_cm <= high_lim)
        freqs_cm, intens = freqs_cm[mask], intens[mask]

        # Apply Gaussian smearing/interpolation
        freqs_cm, intens = get_spectra_curve(xspace, intens, freqs_cm, std)

        if run_no == 0:
            total_intens = intens
        else:
            total_intens += intens
    
    print(f'{total_runs} trajectories used..')
    
    avg_intens = total_intens / total_runs
    return freqs_cm, avg_intens
