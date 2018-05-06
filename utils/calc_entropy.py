def calc_entropy(im, rescale=True):
    
    im = im.flatten()

    #rescale 0 to 100 for binning purposes
    if(rescale):
        im = (im-min(im))/(max(im) - min(im)) * 100
    
    im = im.astype(int)
    min_val = np.min(im)
    max_val = np.max(im)

    vals = np.arange(min_val,max_val+1)
    vals_count = np.zeros_like(vals)

    #add up the occurance of each value
    for px in im:
        vals_count[px] += 1
    #normalize
    vals_prob = vals_count.astype(np.float) / np.sum(vals_count)

    #calc entropy contribution for each instance
    #some values are zero because they have probability zero. Make these zero
    vals_entropy_contrib = -1* vals_prob * np.nan_to_num(np.log(vals_prob)/np.log(2))

    #calc entropy total
    entropy = np.sum(vals_entropy_contrib)

    return(entropy)