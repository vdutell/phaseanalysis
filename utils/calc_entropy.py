def calc_entropy(im, rescale=True):
    
    im = im.flatten()

    #rescale 0 to 100 for binning purposes
    #if(rescale):
    #    im = (im-min(im))/(max(im) - min(im)) * 100
    
    #bin 
    boundaries, counts = np.hist(im)
    centers = (boundaries[1:] + boundaries[:-1])/2

    vals_prob = counts.astype(np.float) / np.sum(counts)

    #calc entropy contribution for each instance
    #some values are zero because they have probability zero. Make these zero
    vals_entropy_contrib = -1* vals_prob * np.nan_to_num(np.log(vals_prob)/np.log(2))

    #calc entropy total
    entropy = np.sum(vals_entropy_contrib)

    return(entropy)