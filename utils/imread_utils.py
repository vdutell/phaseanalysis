import cv2

def readin_im(filename, imfolder):
    im = cv2.imread(imfolder+filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #fix colors 
    return(im)

def cosine_window(image):
    #raised cosyne window on image to avoid border artifacts
    (dim1,dim2) = np.shape(image)
    oneD = np.hanning(dim1)
    oneD = np.tile(oneD,(dim2,1)).T
    twoD = np.hanning(dim2)
    twoD = np.tile(twoD,(dim1,1))
    cosfilter = oneD*(twoD) 
    
    filtered_im = image * cosfilter
    return(filtered_im)

def readpreprc_im(filename, imfolder, mean=True, cosine=False):
    #read image in
    im = readin_im(filename, imfolder)
    # average over color channel
    im = np.mean(im,axis=-1)
    #subtract mean
    if(mean):
        im = im - np.mean(im)
    #take cosine window
    if(cosine):
        im = cosine_window(im)
    
    return(im)