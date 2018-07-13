'''
A set of tools for processing images and directories of images

developed by William Gilpin, 2015--present
https://github.com/williamgilpin/swimpy

Dependencies:
+ scipy
+ numpy
+ scikit-image
Please heed the licenses associated with these dependencies
'''

from numpy import *
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
# from pylab import figure, gca, imshow, subplots_adjust, margins
from pylab import *

from scipy.misc import imresize, toimage
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage import filters
from skimage import img_as_float
# from skimage.measure import label
from scipy.ndimage.measurements import label

import re

################################################################
#####################                      #####################
##################### IMAGE-WISE FUNCTIONS #####################
#####################                      #####################
################################################################



def extrude_image(im0):
    """
    Given some image array, extrude it to
    a full RGBA image in 4D float format: M x M x 4
    """
    
    im = copy(im0)
    
    if max(ravel(im))>1:
        im = im.astype(float)/255.
    if len(im.shape)==2:
        im = im[...,newaxis]
    if im.shape[2]==1:
        im = dstack([im,im,im, ones_like(im)])
        
    return im

def imadjust(im0, threshold=.01):
    '''Automatically rescale the histogram of the image

    im0 : N x M x P np.array
        An RGB or grayscale image, where P is the depth 
        of the image
    
    threshold : float
        The fraction of values saturating the lower and 
        upper portions of the histogram
    threshold : 2-list or tuple
        Manually set the upper and lower clipping bounds
    '''
    if type(threshold) in [tuple,list,ndarray]:
        low_threshold, high_threshold = threshold
    else:
        low_threshold, high_threshold = threshold,threshold
    
    im = copy(im0)
    if max(ravel(im))>1:
        im = im.astype(float)/255.
    
    if len(im.shape)==2:
        im = im[...,newaxis]
    
    rgb_vals = []
    for chan_ind in range(im.shape[2]):
        if chan_ind < 3:
            chan = copy(im[...,chan_ind])

            lo_edge = percentile(ravel(chan), 100.*low_threshold)
            hi_edge = percentile(ravel(chan), 100.*(1.-high_threshold))
            new_chan = (chan - lo_edge)/(hi_edge - lo_edge)
            new_chan[new_chan<0.0]=0.0
            new_chan[new_chan>1.0]=1.0
            im[...,chan_ind] = new_chan
        
    return(im)

def imadd(bgim0, fgim0):
    '''
    Add two RGBA images together, clipping values greater than one
    '''
    
    bgim, fgim = extrude_image(bgim0), extrude_image(fgim0)
    
    rgb_vals = []
    for chan_ind in range(4):
        chan_vals = bgim[:,:,chan_ind] + fgim[:,:,chan_ind]
        chan_vals[chan_vals > 1.0] = 1.0
        rgb_vals.append(chan_vals)
    
    return dstack(rgb_vals)

def imcomplement(im):
    '''Take the complement of an image.
    
    This works for both double and int image types

    im : N x M x P array
        where P is the depth of the image. if P does not exist 
        such as for a black and white image, the code will extrude
        the image automatically
    '''
    
    # There's got to be an easier way to test 
    # the image type:
    if max(ravel(im)) > 1:
        max_val = 255
    else:
        max_val = 1.0
    
    if len(im.shape)<3:
        im = im[...,newaxis]
    
    out_layers = []
    for ind in range(im.shape[-1]):
        if ind < 3:
            new_layer = max_val - im[...,ind]
        else:
            new_layer = im[...,ind]
        out_layers.append(new_layer)
    
    if len(out_layers)==1:
        new_im = out_layers[0]
    else:
        new_im = dstack(out_layers)
    
    return new_im

def bw2color(arr0, cmap_name ='bwr', cmap_style='divergent'):
    '''
    Make a 4D array representing a colored plot
    of a grayscale image
    
    arr0 : N x M np.array
        An array of intensity values
    
    cmap_name : str
        The name of one of the matplotlib default colormaps
    
    cmap_style : str OR 2-tuple
        "divergent" spaces the colormap equally so that 
        zero is in the middle
        "linear" rescales the entire histogram univormly between 0 and 1
        Providing a 2-tuple sets the upper and lower bounds of the colormap
        manually
        
        
    Returns
    -------
    
    arr_clr : N x M x 4 np.array
        Array of RGBA values
    
    '''
    
    arr = copy(arr0)
    
    if type(cmap_style) == tuple:
        cmap_min, cmap_max == tuple
    if cmap_style == 'divergent':
        max_pt = max( [abs(max(ravel(arr))),abs(min(ravel(arr)))] )
        cmap_max = max_pt
        cmap_min = -max_pt
    elif cmap_style =='linear':
        cmap_min = min(ravel(arr))
        cmap_max = max(ravel(arr))
    else:
        cmap_min = min(ravel(arr))
        cmap_max = max(ravel(arr))
    arr = (arr-cmap_min)/(cmap_max - cmap_min)

    cmap = get_cmap(cmap_name)
    arr_clr = cmap(arr)
    
    
    return arr_clr

def bw2trans(im0):
    '''This function remaps white values to transparent pixels
    
    Parameters
    ----------
    im0 : NxNxD 
        Any sort of standard image array
    
    Returns
    -------
    rgb_array : NxNx4
        A float image array including and alpha channel
    
    '''
    im = img_as_float(copy(im0))
    if len(im.shape)==2:
        im = dstack((im[..., newaxis],im[..., newaxis],im[..., newaxis]))
        
    intensity_map = 0.2989*im[...,0] + 0.5870*im[...,1] + 0.1140*im[...,2]
    alphavals = 1.0-double(intensity_map)/max(ravel(intensity_map))

    rgb_array = array([im[...,0], im[...,1], im[...,2], alphavals])
    rgb_array = transpose(rgb_array, (1,2,0))

    return rgb_array

def simple_segment(image0, intensity_threshold0, object_threshold, hole_threshold):
    '''
    The simplest image segmentation method by intensity and size
    Assumes data is light objects on a dark background
    
    Parameters
    ----------

    image0 : NxNx1 or NxNx3 or NxNx4 uint8 array
        This assumes a uint8 image with [0,255] as the image intensity
        range of values

    intensity_threshold0 : int
        The intensity threshold (between 0 and 1)
        If 'auto' is given then otsu thresholding is used
    
    object_threshold : int
        The smallest light objects
    
    hole_threshold : int
        The largest black "holes"

    Returns
    -------

    image : NxNx1 bool array
        The binarized image

    '''
    
    image = img_as_float(copy(image0))
    intensity_threshold = intensity_threshold0

    if intensity_threshold=='auto':
        intensity_threshold = filters.threshold_otsu(image)
    else: 
        intensity_threshold=intensity_threshold0
    
    image[image>intensity_threshold] = 255
    image[image<intensity_threshold] = 0
    
    if image.shape[-1]==3:
        image = image.max(axis=2)

    # if image.shape[-1]==3 or image.shape[-1]==4:
    #     image = image.max(axis=2)
        
    image = remove_small_objects(image.astype(bool),object_threshold)
    image = remove_small_holes(image.astype(bool),hole_threshold)
    
    return image

def keep_biggest_object(bw_im, k=1):
    '''
    Given a boolean array, return the k largest connected
    "True" regions only, ignoring the background (largest)

    Parameters
    ----------

    bw_im : 2xN bool array

    Returns
    -------

    out_im : 2xN bool array

    DEV: this function can be sped up
    '''
    lab_im, ntypes = label(bw_im)
    # lab_im = label(bw_im)             # if the slower skimage label function is used
    # ntypes = max(ravel(lab_im))       # then use these lines instead
    
    if ntypes<(k+1):
        out_im = bw_im
    else:
        out_im = zeros_like(bw_im,dtype=bool)
        sizes = []
        for ind in range(ntypes):
            sizes.append(sum(lab_im==ind))
        ranked_sizes = argsort(sizes)[::-1]
        
        for jj in ranked_sizes[1:k+1]:
            out_im += (lab_im==jj)
    
    return out_im


def overlay_images(im1_0, im2_0, alpha_threshold=1.0):
    '''Overlay two images using alpha channels
    
    im1, im2 : image arrays (uint8 is fine)
    
    alpha_threshold : float
        Determines how much the foreground image takes precedence.
        + If alpha_threshold = 1.0, then the foreground image overlays with
        the expected transparency.
        + If alpha_threshold = 0.0, then all nonzero values of alpha get sent
        to one (fully opaque)
    
    '''

    im1, im2 = copy(im1_0), copy(im2_0)

    if len(im2.shape)==3:
        if im2.shape[2]==4:
            im2[im2[...,3]==0] = 0 

    im2 = bw2trans(im2)

    if im2.shape[0] != im1.shape[0]:
        im2 = imresize(im2, im1.shape)

    if len(im1.shape)==2:
        im1 = array([im1, im1, im1, 255+0*im1])
        im1 = transpose(im1, (1,2,0))

    mask = 1.0 - img_as_float(im2[:,:,3])[...,newaxis]
    mask[mask>alpha_threshold] = 1.0
    overlay_im = (im2 * mask) + (im1 * (1-mask))
    overlay_im = toimage(overlay_im, cmin=0.0, cmax=max(ravel(overlay_im)))
    return overlay_im

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero
    From: http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = hstack([
        linspace(0.0, midpoint, 128, endpoint=False), 
        linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    register_cmap(cmap=newcmap)

    return newcmap


def better_imshow(im, dpi=72):
    '''
    This function is for showing images at their full resolution
    
    im : array
        The image array to be saved
        
    dpi : int
        The desired dots per linear inch
    '''
    orig_size = array(im.shape)
    inches_size = orig_size[:2]/float(dpi)
    figure(figsize=tuple(inches_size))
    
    imshow(im, interpolation='nearest', cmap='gray')

def better_savefig(name, dpi=72, pad=0.0):
    '''
    This function is for saving images without a bounding box and at the proper resolution
        The tiff files produced are huge because compression is not supported py matplotlib
    
    
    name : str
        The string containing the name of the desired save file and its resolution
        
    dpi : int
        The desired dots per linear inch
    
    pad : float
        Add a tiny amount of whitespace if necessary
    
    '''
    gca().set_axis_off()
    subplots_adjust(top = 1+pad, bottom = 0+pad, right = 1+pad, left = 0+pad, 
                hspace = 0, wspace = 0)
    margins(0,0)
    gca().xaxis.set_major_locator(NullLocator())
    gca().yaxis.set_major_locator(NullLocator())

    savefig(name, bbox_inches='tight', pad_inches=0, dpi = dpi)



####################################################################
#####################                          #####################
##################### DIRECTORY-WISE FUNCTIONS #####################
#####################                          #####################
####################################################################


import os
import glob


from skimage import img_as_float

try:
    from skimage.io import imread as imread2 
except ImportError:
    warn('skimage.io not imported, trying slower numpy.imread')
    from numpy import imread as imread2 

def sort_by_numberstr(key):
    '''A sort function based on numbers in the filename string'''
    frag = re.compile(r'(\d+)').split(key)
    frag[1::2] = map(int, frag[1::2])
    return frag

def overlay_images_dir(dir1, dir2, out_dir='', ftype1='.png',ftype2='.png',alpha_threshold=1.0):
    '''
    Given two directories full of images, load one image from each directory and 
    overlay it on the other, respecting the alpha value for transparency

    Parameters
    ----------

    dir1 : str
        Path to the directory of images that will be used as the background

    dir2 : str
        Path to the directory of images that will be tinted and overlaid

    out_dir : str
        Path to the directory at which output will be saved
        
    alpha_threshold : float
        Determines how much the foreground image takes precedence.
        + If alpha_threshold = 1.0, then the foreground image overlays with
        the expected transparency.
        + If alpha_threshold = 0.0, then all nonzero values of alpha get sent
        to one (fully opaque)

    '''
    
    if not out_dir:
        out_dir = os.path.split(dir1)[0]
    
    bg_ims = glob.glob(os.path.join(dir1,'*'+ftype1))
    bg_ims = sorted(bg_ims , key=sort_by_numberstr)

    fg_ims = glob.glob(os.path.join(dir2,'*'+ftype2))
    fg_ims = sorted(fg_ims , key=sort_by_numberstr)

    if len(bg_ims) != len(fg_ims):
        warnings.warn("The two image directories contain different numbers of images.")

        
    for fg_im, bg_im in zip(fg_ims, bg_ims):

        im1 = imread2(bg_im)
        im2 = imread2(fg_im)
        overlay_im = overlay_images(im1, im2, alpha_threshold=alpha_threshold)
    
        bg_name = os.path.split(bg_im)[-1][:-4]
        fg_name = os.path.split(fg_im)[-1][:-4]
        savestr = os.path.join(out_dir, bg_name+'_times_'+fg_name+'.png')
        overlay_im.save(savestr)
