import mkl
mkl.set_num_threads(8)
import numpy as np
from scipy.signal import hilbert
from scipy.signal import hilbert2

import utils.distributions as dists

   
#multithreaded fft support - increases speed for fft & ifft of >100x100 array
try:
    import pyfftw
    import multiprocessing
    nthread = multiprocessing.cpu_count()
    #set default functions as faster pyfftw fft
    def fft2(img):
        return(pyfftw.interfaces.numpy_fft.fft2(img, threads=nthread))
    def ifft2(ft):
        return(pyfftw.interfaces.numpy_fft.ifft2(ft, threads=nthread))
except ImportError:
    print('Couldn\'t find pyfftw or multiprocessing package. Using default fft')
    #set default functions as numpy fft
    def fft2(img):
        return(np.fft.fft2(img))
    def ifft2(ft):
        return(np.fft.ifft2(ft))

               
#measure energy of a 1d function
def get_local_energy_1d(f):
    #f = func - np.mean(func)
    h = np.imag(hilbert(f))
    energy = np.sqrt(np.square(f) + np.square(h))
    return(energy) 

def get_local_pc_1d(f):
    e = get_local_energy_1d(f)
    pc = e/np.mean(np.abs(fft2(f)))
    return(pc)

#measure energy over one axis of an image
def measure_energy_axis(img,axis):
    e = np.apply_along_axis(get_local_energy_1d, axis, img) 
    return(e)

##measure pc over one axis of images
def measure_pc_axis(img, axis):
    e = measure_energy_axis(img,axis)
    pc = e/np.mean(np.abs(np.fft.fft2(img,axis=axis)))
    return(pc)

def measure_energy1_2d(f):
    
    hil = hilbert2(f)
    energy2d = np.sqrt(np.square(np.real(hil))+np.square(np.imag(hil)))
    #phibar = np.arctan(h/f)
    phibar = np.arctan2(np.imag(hil),np.real(hil))
    
    return(energy2d, phibar)


def measure_pc1_2d(img, epsilon=0.001):

    a = np.abs(fft2(img))+epsilon
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

    a = np.abs(fft2(img))+epsilon
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

    
def pc_recon_im(phibar, energy):
    #reconstruct an image from phibar and energy values
    im = np.multiply(np.cos(phibar),energy)
    return(im)


def fft_recon_im(amplitude, phase, real_cast=True):
    #reconstruct an image from amplitude and phase values
    complex_recon = ifft2(np.fft.ifftshift(amplitude*np.exp(1j*phase)))
    if(real_cast):
        return(np.real(complex_recon))
    else:
        return(complex_recon)


def gen_pc(image_dims, mean_pc_goal = 0.1, thresh = 0.01, max_iters = 10000, step_size = 0.01, onef_alpha = 1.2, onef_k = 100000, tradeoff = 1):
    
    '''
    Generate an image patch with 1/f amplitude and a given mean PC value
    
    Parameters:
    image_dims (1x2 float array): Dimensions of image to be generated
    mean_pc_goal (float): Mean PC of ourput image (value to be optimized over)
    thresh (float):  How close to our mean_pc_goal is 'good enough'?
    max_iters:   If we never reach threshold, when should we give up?
    step_size:   What ratio of pi should we step each time?
    onef_alpha:  What is the desired slope of our amplitude of our image?
    onef_k:     What is the offset of our 1/f slope? (adjusts image contrast)
    
    Returns:
    img (2d numpy array): The resulting image patch (may or may not be converged)
    im_seed (2d numpy array): The initial image we started with
    meanpc_evolution (1d numpy array):   The evolution of the PC value over optimization
    '''
    
    #loss function 
    def loss_func(amplitude, angle, pc_goal):
        #trick here use energy for traversing space since it is scalar multiple of PC (and also easier to compute)
        #get complex recon so we can force imaginary component to zero
        recon_complex = fft_recon_im(amplitude, angle, real_cast=False)
        pc = measure_energy_2d(np.real(recon_complex))[0]
        #pc = measure_pc_2d(fft_recon_im(amplitude, phi))[0]
        #cost function is a combo between matching pc and making image real.
        err = np.abs(np.mean(pc) - pc_goal) + tradeoff* np.mean(np.abs(np.imag(recon_complex)))
        #print(err, np.mean(np.abs(np.imag(recon_complex))))
        return(err, recon_complex)
    
    #random 1/f amplitude spectrum
    imamp = dists.make_onef_amp(image_dims, onef_alpha, onef_k)
    
    #annealing vector
    annealing_vec = np.divide(1.,np.sqrt(np.linspace(1,100,num=max_iters)))
    
    #initialized value for phi (fft phase & value x to be perturbed (dx)).
    phi = np.random.rand(*image_dims)*2*np.pi - np.pi
    init_phi = phi
    im_seed = fft_recon_im(imamp, phi)
    #resutling error
    loss = loss_func(imamp,phi,mean_pc_goal)[0]
    #count our iterations
    iters = 0
    #evolution of meanpc
    meanpc_evolution = []
    
    #iterate while our threshold is not reached
    while(loss > thresh):
        #v: random vector we will purturb with
        randphivec = np.random.uniform(-step_size, step_size, size=image_dims)*np.pi
        randphivec = randphivec * annealing_vec[iters]
        tryphi = dists.mod_npi_pi(phi+randphivec)
        tryloss, reon_complex = loss_func(imamp, tryphi, mean_pc_goal)
        if(tryloss < loss):
            phi = tryphi
            loss = tryloss

        else:
            phi = dists.mod_npi_pi(phi - randphivec)
            loss, recon_complex = loss_func(imamp, phi, mean_pc_goal)
        
        #calc new PC value to see if we are at threshold
        pc = measure_pc_2d(np.real(reon_complex))[0]    

        #keep track of evolution
        meanpc_evolution.append(np.mean(pc))
        #check iterations
        iters+=1
        if(iters % 1000 == 0):
            print('*',end='')
            if(iters %10000 == 0):
                print(iters,end='')
        if(iters >= max_iters):
            break
    
    #check if we reached thresh or needed more iterations
    if(iters >= max_iters):
        print(f'\nDidn\'t get to threshold! stopped at {iters} iterations.')
    else:
        print(f'\nReached Threshold in {iters} iterations!')

    #this is our final image - do a real cast here.
    img = fft_recon_im(imamp, phi)
    
    # return our final image
    return(img, imamp, phi, im_seed, init_phi, meanpc_evolution)