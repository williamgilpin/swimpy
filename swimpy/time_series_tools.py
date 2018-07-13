'''
A set of tools for analyzing the vector fields produced by a PIV algorithm

developed by William Gilpin, 2015--present
https://github.com/williamgilpin/swimpy

Dependencies:
+ numpy
+ matplotlib
+ scipy
+ imreg_dft (available on PyPI)
+ + If this is deprecated, use imreg.py by Christoph Gohlke. 
Please heed the license associated with these dependencies

'''

from numpy import *
from pylab import imread
import warnings
import os
import glob
import scipy.ndimage

try:
    import imreg_dft as ird
except ImportError:
    warnings.warn('You don\'t have a valid image registration toolbox installed')

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
    warnings.warn('No multiprocessing module found, running code on single processor only')

from swimpy.process_vfields import rotate_vector


def sliding_data_operation(file_list, out_dir, num_to_group, operation_to_perform=median, **kwargs):
    '''Perform a nonlinear sliding operation on a stack of text files
    
    This code assumes that the entire substack of num_to_group
    data files can be stored simultaneously in RAM
    
    file_list : list of str
        A list of file names, of the kind returned by glob.glob
        
    num_to_group : int
        The number of frames to read into the substack
    
    operation_to_perform : func
        Some function to perform on each element of the substack
        before writing it to output
        
    out_dir : str
        where to write the modified data
    
    kwargs : keyword arguments for operation_to_perform
    
    '''

    xyuvm = loadtxt(file_list[0])

    # preallocate storage array
    stack_shape = xyuvm.shape+(num_to_group,)
    stack = zeros(stack_shape)
    
    
    for ii in range(len(file_list)-num_to_group):
        
        if ii == 0:    
            for jj in range(num_to_group):
                xyuv = loadtxt(file_list[jj])
                stack[...,jj] = xyuv
        else:
            stack = roll(stack, -1, axis=-1)
            xyuv = loadtxt(file_list[ii+num_to_group])
            stack[...,-1] = xyuv

        stack2 = stack.copy()
        
        mod_stack = operation_to_perform(stack2,**kwargs)
        

        txtname = os.path.split(file_list[ii])[-1][:-4]
        param_str = '_'+str(operation_to_perform.__name__)+'_'+str(num_to_group)+'frames'+'.txt'
        savetxt(out_dir+'/'+txtname + param_str, mod_stack, delimiter='\t')


def register_directory(image_dir, ref_im='', out_dir='', save_txt=True,save_image=False,**kwargs):
    '''Run FFT-based rigid image registration comparing images in
        a directory to a reference image
    
    Inputs
    ------
    image_dir : str
        The path to the image files
        
    ref_im : array
        The image to which each image in hte directory will be
        registered. Defaults to the first image in the directory

    out_dir : str
        The path to write the output .txt fles

    save_txt : bool
        Write the affine transformation parameters to a .txt file

    save_image : bool
        save the registered output file
        
    **kwargs : dict
        The arguments for the imreg_dft function
    
    '''
    
    imd = glob.glob(image_dir+'/*.tif')

    imd1 = list(imd)
    imd1.pop(-1)
    
    test_frame = imread2(imd1[0])
    if (len(test_frame.shape) > 2)and (test_frame.shape[-1] > 1):
        rgb_flag = True
        test_frame = test_frame[:,:,0]
        warnings.warn("RGB image detected, first channel will be used.")
    else:
        rgb_flag = False

    if not ref_im:
        ref_im = test_frame

    # text_file = open(out_dir+'/'+"transform_params.txt", "w")
    text_file = open(image_dir+"_transform_params.txt", "w")

    for im1 in imd1:
        frame_a = imread2(im1)
        if rgb_flag:
            frame_a = frame_a[:,:,0]
        
        result = ird.similarity(ref_im, frame_a, **kwargs)
        transform_params = result['scale'], result['angle'], result['tvec']

        if save_image:
            imname = os.path.split(im1)[-1][:-4]
            param_str = '_scale'+str(scale)+'_angle'+str(rotangle)\
                        +'_trans'+str(transvec[0])+'_'+str(transvec[1])
            savestr = out_dir +'/'+imname+'.png'
            toimage(result['timg']).save(savestr)
            pass

        if save_txt:
            print("{0}\t{1}\t{2}".format(*transform_params), file=text_file)

    text_file.close()

def register_directory_pairwise(image_dir, out_dir='', save_txt=True,save_image=False,**kwargs):
    '''Run FFT-based rigid image registration comparing sequential images
        in a directory
    
    image_dir : str
        The path to the image files
        
    out_dir : str
        The path to write the output image fles

    save_txt : bool
        Write the affine transformation parameters to a .txt file
    save_image : bool
        save the registered output file
        
    **kwargs : dict
        The arguments for the imreg_dft function
    
    '''
    
    imd = glob.glob(image_dir+'/*.tif')

    imd1 = list(imd)
    imd2 = list(imd)
    imd1.pop(-1)
    imd2.pop(0)
    
    test_frame = imread2(imd1[0])
    if (len(test_frame.shape) > 2)and (test_frame.shape[-1] > 1):
        rgb_flag = True
        warnings.warn("RGB image detected, first channel will be used.")
    else:
        rgb_flag = False

    text_file = open(image_dir+"_transform_params.txt", "w")

    for im1, im2 in zip(imd1, imd2):
        frame_a, frame_b = imread2(im1), imread2(im2)
        if rgb_flag:
            frame_a, frame_b = frame_a[:,:,0], frame_b[:,:,0]

        result = ird.similarity(frame_a, frame_b, **kwargs)
        transform_params = result['scale'], result['angle'], result['tvec']

        if save_image:
            imname = os.path.split(im1)[-1][:-4]
            param_str = '_scale'+str(scale)+'_angle'+str(rotangle)\
                        +'_trans'+str(transvec[0])+'_'+str(transvec[1])
            savestr = out_dir +'/'+imname+'.png'
            toimage(regim).save(savestr)
            pass

        if save_txt:
            print("{0}\t{1}\t{2}".format(*transform_params), file=text_file)

    text_file.close()

def import_pivdata(filepath, indices=(0,1,2,3), filetype='.txt'):
    '''Import a single PIV dataset and put it in the standard file format, a 2xN
    column vector of x positions, y positions, u components, v components
    
    filepath : str
        A path to the piv data set
    
    indices : tuple of ints
        The indices corresponding to the components that should
        be extracted and returned
        
    filetype : str
        The format of the data (right now only txt supported)
    
    '''
    
    desired_data = [loadtxt(filepath)[:,ind].T for ind in indices]
    return (array(desired_data))

def reshape_pivdata(vec_vals,dimensions=''):
    '''
    Given a list of coordinates or vector values corresponding 
    to PIV data, correctly reshape the array to a square or rectangle.
    If dimensions are not specified, assume a square array
    
    vecvals : 1xM array
        A list of values of x, y, u, or v associated with a PIV data set
    
    dimensions : 2-tuple
        Length and width of PIV data field. If not specified this function
        assumes a square
    '''
    
    if not dimensions:
        xdim = int(sqrt(len(vec_vals)))
        ydim = int(xdim)
    else:
        assert(dimensions[0]*dimensions[1] == len(vec_vals))
        xdim, ydim = dimensions[0], dimensions[1]
        
    out = reshape(vec_vals, (xdim, ydim))
    
    return out
    

def build_data_matrix(data_dir, downsample_files=8, downsample_per_dataset=1,index_bounds=''):
    '''Given a directory of PIV output .txt files in standard 4-column format,
     build a data matrix for machine learning
    
    Inputs
    ------
    
    data_dir : str
        directory containing PIV data sets
        
    downsample_files : int
        take only every 'downsample_files'th file when building matrix
        
    downsample_per_dataset : int
        When importing data rows, downsample each measurement vector
        
    index_bounds : 2-tuple
        The upper and lower indices to use if truncating the training data
    
    '''
    
    data_matrix = list()
    
    all_filepaths = glob.glob(data_dir+'/*.txt')
    if index_bounds:
        all_filepaths = all_filepaths[index_bounds[0]:index_bounds[1]]
    all_filepaths = all_filepaths[::downsample_files]
    
    for filepath in all_filepaths:
        data = import_pivdata(filepath,indices=(2,3))
        data = ravel(data)
        data = data[::downsample_per_dataset]
        data_matrix.append(data)
    
    return array(data_matrix)


   
def lineprofile_pivdata(data_dir, line_coords, slice_indices='', averaging_method='median', show_profile=False, rotate_comps=True):
    '''
    Extract the line profile from a collection of PIV velocity fields in a single
    directory, and then average the results across the entire directory
    
    Because subpixel interpolation is being used, it's best to display these
    as lines not points
    
    data_dir : str
        The path to the directory containing PIV .txt files
    
    line_coords : 2-list of 2-lists
        The coordinates specifying a line of the form
        ((x0,y0), (x1,y1))
    
    slice indices : 3-tuple
        The first, last, and increment slice
    
    averaging_method : str
        The type of averaging method to use:
        median
        mean
    
    show_profile : bool
        Make a drawing showing the values field from one frame with
        the line trace overlaid
        
    rotate_comps : bool
        Project the velocity traces into components parallel and 
        perpendicular to the line trace (parallel, perp)
        
    Returns
    -------
    
    (outui, outvi) : (list, list) 
        A list of interpolated horizontal (ui) and vertical (vi) velocity
        components
    '''
    
    subpixel_factor = 2.0 # factor to interpolate beyond the input pixel density
    
    alldata = glob.glob(data_dir+'/*.txt')
    
    if not slice_indices:
        start,stop,skip = (0,-1,1)
    else:
        start, stop, skip = slice_indices
    alldata = alldata[start:stop][::skip]
        
    ((x0,y0), (x1,y1)) = line_coords
    
    nvals = subpixel_factor*ceil(sqrt((x1-x0)**2 + (y1-y0)**2))
    x, y = linspace(x0, x1, nvals), linspace(y0, y1, nvals)
    
    all_ui, all_vi = list(), list()
    for dataset in alldata:
        (u, v) = import_pivdata(dataset,indices=(2,3))
        u = reshape_pivdata(u)
        v = reshape_pivdata(v)
        all_ui.append( scipy.ndimage.map_coordinates(u, vstack((x,y))) )
        all_vi.append( scipy.ndimage.map_coordinates(v, vstack((x,y))) )
    
    outui, outvi = median(array(all_ui),axis=0), mean(array(all_vi),axis=0)
    
    if rotate_comps:
        rotang = pi-arctan2(x1-x0, y1-y0)
        all_rots = rotate_vector(array(list(zip(outui, outvi))), rotang, deg=False)
        outui, outvi = all_rots[:,0], all_rots[:,1]
    
    if show_profile:
        figure()
        #imshow(u,origin='lower')
        imshow(log(sqrt(u**2 + v**2)),origin='lower')
        hold(True)
        plot([x0,x1], [y0,y1])
        
        plot(linspace(x0,x1,5),linspace(y0,y1,5),'.')
        
        #org_vec = rotate_vector(array([x0,y0]), -rotang, deg=False)
        #end_vec = rotate_vector(array([x1,y1]), -rotang, deg=False)
        #plot([org_vec[0],end_vec[0]], [org_vec[1],end_vec[1]])
    
    return (outui, outvi)