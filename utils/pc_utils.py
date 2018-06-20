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

def measure_energy1_2d(f):
    
    hil = hilbert2(f)
    energy2d = np.sqrt(np.square(np.real(hil))+np.square(np.imag(hil)))
    #phibar = np.arctan(h/f)
    phibar = np.arctan2(np.imag(hil),np.real(hil))
    
    return(energy2d, phibar)


def measure_pc1_2d(img, epsilon=0.001):

    a = np.abs(np.fft.fft2(img))+epsilon
    e, phibar = measure_energy1_2d(img)
    pc_vals = np.divide(e,np.mean(a))

    return(pc_vals, phibar)
    
def measure_energy2_2d(f):

    h = np.imag(hilbert2(f))
    #phibar = np.arctan(h/f)
    phibar = np.arctan2(h,f)
    energy2d = np.multiply((np.cos(phibar) - np.abs(np.sin(phibar))), f) 

    return(energy2d, phibar)

    
def measure_pc2_2d(img, epsilon=0.001):

    a = np.abs(np.fft.fft2(img))+epsilon
    e, phibar = measure_energy2_2d(img)
    pc_vals = np.divide(e,np.sum(a))
    
    return(pc_vals, phibar)

    
def measure_pc_2d(img, pcn = 1, pbflag=True, epsilon= 0.001):
    # wrapper function to take care of two different pc measurement types
    
    if(pcn==1):
        pc_vals, phibar = measure_pc1_2d(img, epsilon)
    elif(pcn==2):
        pc_vals, phibar = measure_pc2_2d(img, epsilon)

    if(pbflag==False):
        return(pc_vals)
    else:
        return(pc_vals, phibar)
    
def measure_energy_2d(img, en=1):
    # wrapper function to take care of two different energy measurement types
    if(en==1):
        e_vals, phibar = measure_energy1_2d(img)
    elif(en==2):
        e_vals, phibar = measure_energy2_2d(img)
    return(e_vals, phibar)

    
def pc_recon_im(phibar, e):
    #reconstruct an image from phibar and energy values
    im = np.multiply(np.cos(phibar),e)
    return(im)


def fft_recon_im(amp, phase):
    recon = np.real(np.fft.ifft2(np.fft.ifftshift(amp*np.exp(1j*phase))))
    return(recon)
        
    