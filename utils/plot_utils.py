import matplotlib.pyplot as plt
import utils.pc_utils as pcu

import numpy as np

plt.rcParams['figure.figsize'] = [16, 16]

def compare_initim_genim(initim, genim, alpha, show=True):
    
    #plot init im
    plt.subplot(121)
    plt.imshow(initim, cmap='Greys_r')
    plt.axis('off')
    plt.title(f'Initial: alpha={alpha}')
    plt.subplot(122)
    #plot generated im
    plt.imshow(genim,cmap='Greys_r')
    plt.axis('off')
    plt.title(f'Generated: alpha={alpha}')
    plt.savefig(f'output/initial_generated_ims_alpha{alpha}.png',dpi=800)
    if(show):
        plt.show()
    
def pc_evolution(meanpc_evolution, alpha, show=True): 
    plt.figure(figsize=(10,10))
    plt.plot(meanpc_evolution)
    plt.title('Evolution of Mean PC')
    plt.savefig(f'output/evolution_alpha{alpha}.png',dpi=800)
    if(show):
        plt.show()
    
def compare_initim_genim_stats(initim, genim, alpha, show=True):
    plt.figure(figsize=(10,4))
    ipc, ipb = pcu.measure_pc_2d(initim)
    ift = np.fft.fftshift(np.fft.fft2(initim))
    iap = np.abs(ift)
    igp = np.angle(ift)
    plt.subplot(251)
    plt.imshow(initim,cmap='Greys_r')
    plt.axis('off')
    plt.title('Init Im')
    plt.subplot(252)
    plt.imshow(ipc,cmap='Greys_r')
    plt.axis('off')
    plt.title('Init PC')
    plt.subplot(253)
    plt.imshow(ipb,cmap='hsv')
    plt.title('Init PhiB')
    plt.axis('off')
    plt.subplot(254)
    plt.imshow(iap,cmap='Greys_r')
    plt.title('Init Amp')
    plt.axis('off')
    plt.subplot(255)
    plt.imshow(igp,cmap='hsv')
    plt.title('Init GP')
    plt.axis('off')

    gpc, gpb = pcu.measure_pc_2d(genim)
    gft = np.fft.fftshift(np.fft.fft2(genim))
    gap = np.abs(gft)
    ggp = np.angle(gft)
    plt.subplot(256)
    plt.imshow(genim,cmap='Greys_r')
    plt.title('Gen Im')
    plt.axis('off')
    plt.subplot(257)
    plt.title('Gen PC')
    plt.imshow(gpc,cmap='Greys_r')
    plt.axis('off')
    plt.subplot(258)
    plt.title('Gen PB')
    plt.imshow(gpb,cmap='hsv')
    plt.axis('off')
    plt.subplot(259)
    plt.title('Gen Amp')
    plt.imshow(gap,cmap='Greys_r')
    plt.axis('off')
    plt.subplot(2,5,10)
    plt.title('Gen GP')
    plt.imshow(ggp,cmap='hsv')
    plt.axis('off')
    plt.savefig(f'output/initial_generated_pcpb_alpha{alpha}.png', 
                    dpi=800)
    if(show):
        plt.show()

def hist_pc_dists(initim,genim,cats,trail,beach, alpha, show=True):
    plt.figure(figsize=(10,10))
    aval=0.2
    plt.hist(pcu.measure_pc_2d(initim)[0].flatten(),bins=100,normed=True,label='initial', alpha = aval);
    plt.hist(pcu.measure_pc_2d(genim)[0].flatten(),bins=100,normed=True,label='generated', alpha = aval);
    plt.hist(pcu.measure_pc_2d(cats)[0].flatten(),bins=100,normed=True,label='cats', alpha = aval);
    plt.hist(pcu.measure_pc_2d(trail)[0].flatten(),bins=100,normed=True,label='trail', alpha = aval);
    plt.hist(pcu.measure_pc_2d(beach)[0].flatten(),bins=100,normed=True,label='beach', log=True,alpha = aval);
    plt.title('Distribution of Pixelwise PC')
    plt.legend()
    plt.savefig(f'output/initial_generated_dists_alpha{alpha}.png',dpi=800)
    if(show):
        plt.show()