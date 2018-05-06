import numpy as np
from scipy.signal import hilbert

#measure energy of a 1d function
def get_local_energy_1d(func):
    f = func - np.mean(func)
    h = np.imag(hilbert(f))
    energy = np.sqrt(f**2 + h**2)
    return(energy) 

#measure energy over one axis of an image
def measure_energy_axis(img,axis):
    e = np.apply_along_axis(get_local_energy_1d, axis, img) 
    return(e)

#measure energy over an image
def measure_energy_2d(img):
    
    x_e = measure_energy_axis(img,axis=0)
    y_e = measure_energy_axis(img,axis=1)
    
    e_vals = np.sum([x_e,y_e],axis=0)
    
    return(e_vals)

#measure phase congruency
def measure_pc_2d(img):
    
    #can't just divide x and y individually.
    e_vals = measure_energy_2d(img)
    
    pc_vals = e_vals / np.mean(np.abs(np.fft.fft2(img)))
    
    return(pc_vals)
