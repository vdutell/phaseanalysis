import numpy as np

def radial_analysis(signal2d, func, center=None):
    """
    Calculate function of a signal for all values at a given radius.

    signal2d - 2d signal to be analyzed (not necessarily an signal_2ds)
    func - function to be applied to signal radially
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the signal (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the signal_2ds
    y, x = np.indices(signal2d.shape)

    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = signal2d.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = (r_sorted/1.).astype(int)

    rads = np.unique(r_int)
    vals= []
    for rad in rads:
        pxls = i_sorted[np.where(r_int == rad)]
        vals.append(func(pxls))

    #remove values larger than largest with full circle
    maxind = np.min((signal2d.shape[0]//2,signal2d.shape[1]//2))
    rads = rads[:maxind]
    vals = vals[:maxind]

    
    return(rads, vals)