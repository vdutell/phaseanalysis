import numpy as np
from scipy.signal import hilbert
from scipy.signal import hilbert2

#measure energy of a 1d function
def get_local_energy_1d(f):
    #f = func - np.mean(func)
    h = np.imag(hilbert(f))
    energy = np.sqrt(np.square(f) + np.square(h))
    return(energy) 

#measure energy over one axis of an image
def measure_energy_axis(img,axis):
    e = np.apply_along_axis(get_local_energy_1d, axis, img) 
    return(e)

##measure pc over one axis of images
def measure_pc_axis(img, axis):
    e = measure_energy_axis(img,axis)
    pc = e/np.mean(np.abs(np.fft.fft(img,axis=axis)))
    return(pc)

#measure energy over an image
def measure_energy_2d(f):
    
    #x_e = measure_energy_axis(img,axis=0)
    #y_e = measure_energy_axis(img,axis=1)
    #energy2d = np.sum([x_e,y_e],axis=0)
    
    energy2d = np.sqrt(np.square(f)+np.square(np.imag(hilbert2(f))))
    
    return(energy2d)

#measure phase congruency
def measure_pc_2d(img):
    
    #can't just divide x and y individually.
    e_vals = measure_energy_2d(img, epsilon=0.01)
    a = np.mean(np.abs(np.fft.fft2(img)))+epsilon
    pc_vals = np.divide(e_vals, a)
    
    return(pc_vals)
