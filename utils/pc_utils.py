import numpy as np
from scipy.signal import hilbert
from scipy.signal import hilbert2

import utils.distributions as dists


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


def gen_pc(image_dims, mean_pc_goal = 0.1, thresh = 0.01, max_iters = 10000, step_size = 0.01, onef_alpha = 1.2, onef_k = 100000):
    
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
    def loss_func(amp, phi, pc_goal):
        #trick here use energy for traversing space since it is scalar multiple of PC (and also easier to compute)
        pc = measure_energy_2d(fft_recon_im(amp, phi))[0]
        #pc = measure_pc_2d(fft_recon_im(amp, phi))[0]
        err = np.abs(np.mean(pc) - pc_goal)
        return(err)
    
    #random 1/f amplitude spectrum
    amp = dists.make_onef_amp(image_dims, onef_alpha, onef_k)
    
    #annealing vector
    annealing_vec = np.divide(1.,np.sqrt(np.linspace(1,1000,num=max_iters)))
    
    #initialized value for phi (fft phase & value x to be perturbed (dx)).
    phi = np.random.rand(*image_dims)*2*np.pi - np.pi
    im_seed = fft_recon_im(amp,phi)
    #resutling error
    loss = loss_func(amp,phi,mean_pc_goal)
    #count our iterations
    iters = 0
    #evolution of meanpc
    meanpc_evolution = []
    
    #iterate while our threshold is not reached
    while(loss > thresh):
        #v: random vector we will purturb with
        randphivec = np.random.uniform(-step_size, step_size, size=image_dims)*np.pi
        randphivec = randphivec * annealing_vec[iters]
        newloss = loss_func(amp, dists.mod_npi_pi(phi+randphivec), mean_pc_goal)
        if(newloss < loss):
            phi = dists.mod_npi_pi(phi + randphivec)
            loss = newloss
        else:
            phi = dists.mod_npi_pi(phi - randphivec)
            loss = loss_func(amp, phi, mean_pc_goal)
        #calc new PC value to see if we are at threshold
        pc = measure_pc_2d(fft_recon_im(amp, phi))[0]
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
        print(f'\nReached Threshold in {iters} inerations!')

    #this is our final image
    img = fft_recon_im(amp,phi)
    
    # return our final image
    return(img, im_seed, meanpc_evolution)