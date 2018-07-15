'''
NOTE: This set of functions has only been tested on Python 2.7+, however the 
latest versions of openpiv support Python 3. I have not yest tested this module
with the newest openpiv version

A set of tools for performing PIV and other operations on directories of images
representing time series

developed by William Gilpin, 2015--present
https://github.com/williamgilpin/swimpy

Dependencies:
+ openpiv-python (http://www.openpiv.net/openpiv-python/)
Please heed the licenses associated with these dependencies

'''

from numpy import *
from pylab import imread
import warnings
import os
import glob

import openpiv.tools
import openpiv.process
import openpiv.scaling

try:
    from scipy.misc import imread as imread2 
except ImportError:
    warn('scipy.misc not imported, using slower numpy.imread')
    from numpy import imread as imread2 

try:
    import multiprocessing as mp
    parallel_available = True
except ImportError:
    parallel_available = False
    warnings.warn('No multiprocessing module found, code will run on single processor only')


def piv_pair(frame_a, frame_b, winsize=60, overlap=30,sch_size=100):
    '''
    Returns
    -------
    These are packed into a tuple (x ,y ,u, v)
    
    x : int array
        An array of locations of the vectors
        
    y : int array
        An array of y locations of the vectors
        
    u : double array
        The x components of the velocity field
        
    v : double array
        The y components of the velocity field
    '''
    frame_a = frame_a.astype('int32')
    frame_b = frame_b.astype('int32')

    u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=winsize,
                                                               overlap=overlap, dt=0.02, search_area_size=sch_size, 
                                                               sig2noise_method='peak2peak' )

    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=winsize, overlap=overlap)

    u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.5 )

    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)

    return x,y,u,v,mask


def piv_directory(image_dir , out_dir, **kwargs):
    '''Run PIV on a time series
    
    image_dir : str
        The path to the image files
        
    out_dir : str
        The path to write the output .txt fles
        
    **kwargs : dict
        The arguments for the PIV function
    
    '''

    imd = glob.glob(image_dir+'\*.tif')

    imd1 = list(imd)
    imd2 = list(imd)
    imd1.pop(-1)
    imd2.pop(0)
    for im1, im2 in zip(imd1, imd2):
        frame_a = imread2(im1)
        frame_b = imread2(im2)
        # xyuv = piv_pair(frame_a, frame_b, winsize=60, overlap=30,search_size=100)
        xyuv = piv_pair(frame_a, frame_b, **kwargs)
        x,y,u,v,mask = xyuv

        imname = os.path.split(im1)[-1][:-4]

        param_str = ''

        if 'winsize' in kwargs:
            param_str += '_window' +str(kwargs['winsize'])
        if 'overlap' in kwargs:
            param_str += '_overlap' +str(kwargs['overlap'])
        if 'sch_size' in kwargs:
            param_str += '_sch' +str(kwargs['sch_size'])

        openpiv.tools.save(x, y, u, v, mask, out_dir+'/'+imname + param_str + '.txt' )
 