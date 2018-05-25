import numpy as np

def rescale(array):
    #rescale between 0 and 1
    array = array - np.min(array)
    array = array / np.max(array)
    return(array)

def rmdc(array):
    #remove DC component
    return(array-np.mean(array))

def make_onef_amp(shape, alpha, k):
    #make an amplitude spectrum of 1/f
    y, x = np.indices(shape)
    center = np.array(shape)/2
    r = np.hypot(x - center[1], y - center[0])
    amp = k/(1+r**alpha)
    return amp

def make_onef_ims(shape, alpha=1, k=1):
    #make an image with 1/f amplitude and uniformly random phase
    amp_onef = make_onef_amp(shape, alpha, k) #get amplitude spectrum
    random_phase = np.random.rand(*shape)*2*np.pi - np.pi #get uniformly random phase dist [-pi,pi]
    recon_onef_rand = np.fft.ifft2(np.fft.ifftshift(amp_onef*np.exp(1j*random_phase))) #inverse fft to get image 
    recon_onef_rand = np.real(recon_onef_rand) #real part of image so we can plot and analyze it.
    recon_onef_rand = rescale(recon_onef_rand) #rescale to 0,1
    return(recon_onef_rand)
