'''
A set of tools for analyzing extracting contour flow information from images
and PIV data set

developed by William Gilpin, 2016--present
https://github.com/williamgilpin/swimpy

Dependencies:
+ scipy
+ numpy
+ scikit-image
+ time_series_tools.py
Please heed the licenses associated with these dependencies 
'''

from numpy import *
from numpy.random import RandomState
from pylab import *
import warnings
import os
import glob

import swimpy.plt_fmt
from swimpy.time_series_tools import import_pivdata, reshape_pivdata
from swimpy.process_images import keep_biggest_object, simple_segment

from scipy import interpolate
from scipy.misc import imresize
from scipy.signal import savgol_filter
from scipy.signal.signaltools import correlate
from scipy.spatial.distance import pdist,squareform

from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries
from skimage.measure import label

try:
    from skimage.io import imread as imread2 
except ImportError:
    warn('skimage.io not imported, trying slower numpy.imread')
    from numpy import imread as imread2 


try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn import decomposition
    has_sklearn=True
except:
    warnings.warn('scikit-learn installation not found. Dimensionality reduction functions will not work.')
    has_sklearn=False




def order_contour(boundary_coords, direction='cw'):
    '''
    Given an unsorted list of coordinates denoting points 
    along a simple boundary, order them so that the coordinates 
    smoothly parametrize the boundary
    
    Intended to recreate the functionality of MATLAB's bwtraceboundary()
    
    Parameters
    ----------
    boundary_coords : 2xN array
        An array of coordinates comprising a 1-dimensional contour
    
    direction : str
        The handedness of the ordering: 'cw' or 'ccw'

    Returns
    -------
    boundary_coords : 2xN array
        An ordered array of coordinates along a 1-dimensional contour

    '''
    
    all_dists = squareform(pdist(boundary_coords))
    
    if len(all_dists)<1:
        warnings.warn("Empty boundary coordinates list found. Check segmentation")

    # Find the index of the topmost point
    start = argmax(boundary_coords[:,1]) 
    pts = [start]
    
    nxt = (item for item in argsort(all_dists[start,:]) if item not in pts)
    if direction=='cw':
        nxt = (item for item in nxt if boundary_coords[item][0] > boundary_coords[start][0])
    elif direction=='ccw':
        nxt = (item for item in nxt if boundary_coords[item][0] < boundary_coords[start][0])
    try:
        nxt = nxt.send(None)
    except StopIteration:
        nxt = start

    # Now order the contour starting at that point
    while len(pts) < len(all_dists):

        nxt = (item for item in argsort(all_dists[nxt,:]) if item not in pts) #lol list comprehensions
        nxt = nxt.send(None)

        pts.append(nxt)
        
    boundary_coords = boundary_coords[pts]
    theta_list = [arctan2(*thing) for thing in boundary_coords]
    coord_sign = sign(median(diff(theta_list)))

    if coord_sign==-1:
        boundary_coords = boundary_coords[::-1]
    
    # # weaker implementation based on angle sorting
    # centroid = array(mean(boundary_coords[1:,0]),mean(boundary_coords[1:,1]))
    # angs = array([arctan2(*item) for item in boundary_coords-centroid])
    # sort_inds = argsort(angs)
    # boundary_coords = boundary_coords[sort_inds]

    return boundary_coords



def contour_flow(mask, u, v, output='vtan'):
    '''
    Given a PIV field and a list of coordinates, compute the flow
    along and across the boundary
    
    Parameters
    ----------
    mask : N x M bool array
        A mask with True denoting locations in an object
        of interest.
    
    u : N' x M' array
        The x components of velocity for a vector field in
        squared shape
        
    v : N' x M' array
        The y components of velocity for a vector field in
        squared shape
    
    output : str
        The quantity to compute on the boundary. Default is the tangential
        flow velocity component.
        'vabs' : return just the total flow speed sampled at the boundary
        points
        
    Returns
    -------
    
    out : 3 x N array
        An array of (x,y,vtan) values for each point on the boundary
    
    '''

    boundary_im = find_boundaries(label(mask))
    boundary_im = skeletonize(boundary_im)
    boundary_coords = argwhere(boundary_im)
    boundary_coords = order_contour(boundary_coords)

    u_up, v_up = imresize(u, mask.shape), imresize(v, mask.shape)
    
    vtan = list()
    vnorm = list()

    for prev_coord, coord in zip(boundary_coords[:-1],boundary_coords[1:]):
        u_val = u_up[tuple(coord)]
        v_val = v_up[tuple(coord)]

        tan_vec = coord - prev_coord
        tan_vec = tan_vec/sqrt(sum(tan_vec**2))
        
        vtan_mag = u_val*tan_vec[0] + v_val*tan_vec[1]
        vtan.append(vtan_mag)
        
        vnorm_mag = array([u_val, v_val]) - vtan_mag*array(tan_vec)
        vnorm_mag = sqrt(sum(vnorm_mag**2))
        vnorm.append(vnorm_mag)
        
    vtan, vnorm = array(vtan), array(vnorm)  
    
    out = hstack((boundary_coords[1:,:], vtan[...,None], vnorm[...,None]))
    
    return out




def velocity_trace_directory(image_dir, piv_dir, *args, out_dir='', segment_images=True):
    '''
    Given (1) the locations of a bunch of images, and (2) the locations of a 
    bunch of PIV data sets for those images, go through the two directories
    and extract the boundary coordinates
    
    Parameters
    ----------
    image_dir : str
        Directory of images to segment and used to extract the boundary profile
    
    out_dir : str
        Directiory to save .txt files containing boundary information

    segment_images : bool
        Segment images before extracting the boundary

    args : the arguments to pass to the segmentation function
            'simple_segment.py'

    
    '''
    
    if not out_dir:
        out_dir = image_dir

    all_images = glob.glob(image_dir+'/*.tif')
    all_pivfiles = glob.glob(piv_dir+'/*.txt')
    
    paired_files = zip(all_images[:len(all_pivfiles)], all_pivfiles)
    
    for image_path, piv_path in paired_files:
        
        # could implement a filename-checker here in case of
        # bad sorting (or use a clever pre-sort step somewhere
        # out of the for loop)
        
        # name = piv_path.split('/')
        _, name = os.path.split(image_path)

        name = name[:-4]
        
        im = imread(image_path, as_grey=True)
        
        if segment_images:
            seg_im = simple_segment(im, *args)
            seg_im = keep_biggest_object(seg_im)
        else:
            seg_im = im

        
        (u, v) = import_pivdata(piv_path,indices=(2,3))
        u = reshape_pivdata(u)
        v = reshape_pivdata(v)

        vel_profile = contour_flow(seg_im, u, v)
#         vel_profile.dump(out_dir + name +'boundary.pkl')
        savetxt( os.path.join(out_dir, name +'_boundary.txt'), vel_profile)


def plot_boundary_contour(vel_profile, do_save=True, name_str ='boundary', save_type='pdf', axlim=''):
    '''
    Given an array of four-dimensional coordinates specifying points along a contour,
    and two components of velocity at those points, plot the first component as a colormap
    on the contour and save an image.
    
    Parameters
    ----------
    do_save : bool
        whether to save image after plotting
    
    save_type : str
        The type of image file to save
        
    axlim : ((x0,xf),(y0,yf))
    
    Dev:
    ----
    + Try OrGr colormap?
    '''

    if not axlim:
        axlim =  ( (.95*min(vel_profile[:,1]), 1.05*max(vel_profile[:,1])),
                    (.95*min(vel_profile[:,0]), 1.05*max(vel_profile[:,0])) )

    fig = figure()
    cm = get_cmap('RdBu')
    cmap_up = max(abs(vel_profile[:,2]))
    cmap_bot = -cmap_up
    # cmap_up = max(vel_profile[:,2])
    # cmap_bot = min(vel_profile[:,2])

    # SG: 11, 1
    scatter(vel_profile[:,1], vel_profile[:,0], c=savgol_filter(vel_profile[:,2], 31, 3),
     vmin=cmap_bot, vmax=cmap_up, s=20.0, cmap=cm, edgecolors='face')
    
    xlim(list(axlim[0]))
    ylim( [list(axlim[1])[1], list(axlim[1])[0]] )

    # xscale, yscale = im.shape
    # even_asp = (yscale)/(xscale)
    # gca().axes.set_aspect(1/even_asp, adjustable='box')

    gca().axes.set_aspect('equal', adjustable='box')
    gca().set_axis_off()
    subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    margins(0,0)
    gca().xaxis.set_major_locator(NullLocator())
    gca().yaxis.set_major_locator(NullLocator())
    
    if do_save:
        savefig(name_str+'.' + save_type, bbox_inches='tight', pad_inches=0, dpi=600)
        close(fig)


def heaviside(x):
    return 0.5*(sign(x)+1)

def center_signal(sig,shift=0.0):
    '''
    Given a signal on a periodic domain with a dominant 
    sine component, remove the phase shift from the signal 
    and return it.
    
    DEV: This function seems to cause the direction to randomly
    switch sides
    '''
    theta_vals = linspace(0, 2*pi, len(sig))
    sig0 = sin(theta_vals)
    sig0 = heaviside(theta_vals-pi)
    mdpt = int(len(sig)/2)

    max_ind = argmax(abs(correlate(sig, sig0)))
    max_ind = mod(max_ind, len(sig))
    
    out_sig = hstack((sig[max_ind:], sig[:max_ind]))
    
    # this is a pretty shaky way to get this phase shift
    if sum(out_sig[:mdpt]) > sum(out_sig[mdpt:]):
        out_sig = hstack((out_sig[mdpt:], out_sig[:mdpt]))

    return out_sig

def find_components(input_data, n_components=2, method='pca',**kwargs):
    '''
    Extract components from an array of data
    
    input_data: np.array
        The input data matrix
    
    n_comps : int
        The number of components to extract
    
    method : str
        The dimensionality reduction technique to use
        
    kwargs : optional arguments to pass to the construction of the estimator
        
    Note: this function is basically a wrapper for a bunch of the 
    standard estimators found in sklearn. Please refer to the sklearn documentation
    for the keyword arguments to pass to the various estimators.
    http://scikit-learn.org/stable/modules/decomposition.html
    
    Examples
    --------
    
    >>>components = find_components(data,method='k-means',tol=1e-3, batch_size=100, max_iter=50)
    >>>plot(components.T)
    
    >>>components = find_components(copy(all_traces.T),method='incremental pca',batch_size=100)
    >>>plot(components.T)
    
    DEV:
     Automatically compute the batch sizes for incremental and minibatch PCA
    "The computational overhead of each SVD is O(batch_size * n_features ** 2), 
    but only 2 * batch_size samples remain in memory at a time. There will be n_samples / batch_size SVD 
    computations to get the principal components, versus 1 large SVD of complexity O(n_samples * n_features ** 2) 
    for PCA."
    
    Issues
    ------

    LDA returns components that are not orthogonal; need to apply Gram-Schmidt
    
    Kernel PCA is currently broken, also LDA is erratic
    http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA
    

    '''
    
    if not has_sklearn:
        warnings.warn('scikit-learn not found. This function will not work correctly.')
        return None
    
    n_samples, n_features = input_data.shape
    rng = RandomState(0)
    
    if n_samples < n_features:
        warnings.warn('More features than samples; assuming input data matrix was transposed.')
        input_data = input_data.T
        n_samples, n_features = n_features, n_samples
    
    if method in ['pca','ica','k-means', 'incremental pca','kernel pca','random pca']:
        data = input_data - input_data.mean(axis=0)
    else:
        data = input_data
    
    if method=='pca':
        estimator = decomposition.PCA(n_components=n_components,**kwargs)
    elif method=='k-means':
        estimator = MiniBatchKMeans(n_clusters=n_components, random_state=rng,**kwargs)
    elif method=='ica':
        estimator = decomposition.FastICA(n_components=n_components, whiten=True,**kwargs)
    elif method=='incremental pca':
        estimator = decomposition.IncrementalPCA(n_components=n_components, whiten=True)
    elif method=='svd':
        estimator = decomposition.TruncatedSVD(n_components=n_components, random_state=rng,**kwargs)
    elif method=='kernel pca':
        estimator = decomposition.KernelPCA(n_components=n_components,**kwargs)
    elif method=='lda':
        estimator = decomposition.LatentDirichletAllocation(n_topics=n_components,random_state=rng,**kwargs)
        data = data + abs(min(ravel(data)))
    elif method=='random pca':
        estimator = decomposition.RandomizedPCA(n_components=n_components, whiten=True,**kwargs)
    else:
        warnings.warn('Unknown \'method\' argument given; falling back to PCA')
        estimator = decomposition.PCA(n_components=n_components)

    estimator.fit(data)
    
    if hasattr(estimator, 'cluster_centers_'):
        components = estimator.cluster_centers_
    else:
        components = estimator.components_
    

    return components





####################################################################################################
####################################################################################################
###################                                                              ###################
###################         Tools for Regularizing and interpolating data        ###################
###################                                                              ###################
####################################################################################################
####################################################################################################



def interpolate_array(data, nbins=360):
    '''
    Interpolate an array onto one with a given number of bins

    data : array
        The original data set

    nbins  int
        The number of points in the target array
    '''
   
    xvals_interp = linspace(0,1, nbins)
        
    xvals = linspace(0,1,len(data))
    f = interpolate.interp1d(xvals, data)
    data_interp = f(xvals_interp)
    
    return data_interp
    


def interp_time_series(array_list, upsample_factor=1):
    '''
    Given time series (list) containing arrays of values
    corresponding to line traces, interpolate the data
    onto a uniform mesh so that the dataset can be represented
    as an array.

    Parameters
    ----------
    array_list : list
        A list of arrays of different lengths

    upsample_factor : float
        The amount to upsample the data beyond the number 
        of values in the first array


    Returns
    -------
    out : array
        An array of interpolated values
    
    '''
    
    all_traces = []
    for trace in array_list:
        if len(all_traces)==0:
            N = int(upsample_factor*len(trace))
            xvals_interp = linspace(0,1,N)

        # trace_interp = interpolate_array(trace, nbins=N)
        
        xvals = linspace(0,1,len(trace))
        f = interpolate.interp1d(xvals, trace)
        trace_interp = f(xvals_interp)
        all_traces.append(trace_interp)
    
    out = array(all_traces)
    
    return out


def interpolate_nans(arr):
    '''
    Given a numpy array containing some NaNs, interpolate 
    the missing values
    
    Note: for large amounts of missing data, inpainting should
    be used instead of interpolation
    '''
    
    y = copy(arr)
    nans_bool = isnan(y)
    
    x = lambda z: z.nonzero()[0]
    
    nan_pts = nans_bool.nonzero()[0]
    not_nan_pts = (~nans_bool).nonzero()[0]
    
    y[nans_bool]= interp(nan_pts, not_nan_pts, y[~nans_bool])
    
    return y

def interpolate_array(data, nbins=360):
    '''
    Interpolate an array onto one with a given number of bins

    data : array
        The original data set

    nbins  int
        The number of points in the target array
    '''
   
    xvals_interp = linspace(0,1, nbins)
        
    xvals = linspace(0,1,len(data))
    f = interpolate.interp1d(xvals, data)
    data_interp = f(xvals_interp)
    
    return data_interp


def interpolate_contour(velprofile, nbins=360, method='direct'):
    '''
    Given a four-column profile containing x,y,vperp,vpar,
    interpolate to a contour with a fixed number of bins
    
    Parameters
    ----------
    
    velprofile_vals : N x 4 array
        A four-column velocity profile of the form x,y,vperp,vpar
        representing:
        x coordinates of points on boundary, 
        y coordinates of points on boundary,
        parallel velocity value of points on boundary,
        perpendical velocity values at points on boundary
    
    nbins : int
        The number of angular bins to use
        
    method : str
        The type of interpolation to use:
        'direct' preserves length
        'circular' averages along each angular bin relative to centroid
        DEV: 'curvature' rescales by the local curvature
        
    Returns
    -------
    
    out : N x 3 array
        The boundary coordinate, average parallel, and perpendicular velocity component
        
    
    '''
    
    if method not in ['direct', 'circular']:
        method = 'direct'
    
    if method=='direct':
        bin_vals = arange(1,nbins+1)
        interp_par = interpolate_array(vel_profile[:,2], nbins=360)
        interp_perp = interpolate_array(vel_profile[:,3], nbins=360)
        out = vstack([bin_vals, interp_par, interp_perp]).T
    elif method=='circular':
        out = interpolate_contour_radial_average(velprofile, 
                                                 nbins=nbins, fill_nans=True)
    else:
        warnings.warn('Selection of interpolation method failed.')
        
    return out

    

def interpolate_contour_radial_average(velprofile_vals, nbins=360, fill_nans=True):
    '''
    Given a four-column profile containing x,y,vperp,vpar,
    find the average component of vperp,vpar along each angular direction
    
    Parameters
    ----------
    
    velprofile_vals : N x 4 array
        A four-column velocity profile of the form x,y,vperp,vpar
        representing:
        x coordinates of points on boundary, 
        y coordinates of points on boundary,
        parallel velocity value of points on boundary,
        perpendical velocity values at points on boundary
    
    nbins : int
        The number of angular bins to use
        
    fill_nans : bool
        If values are missing, interpolate them
    
    Returns
    -------
    
    out : N x 3 array
        The averaged parallel and perpendicular velocity component
        at each theta value
    
    BUGS: if nbins is too large, interpolating nans may fail
    '''
    nbins = 360
    theta_vals = arctan2(velprofile_vals[:,1]-median(velprofile_vals[:,1]),
                         velprofile_vals[:,0]-median(velprofile_vals[:,0]))    
    count_vals, bin_vals = histogram(theta_vals, bins=nbins)
    ind_vals = digitize(theta_vals, bin_vals)
    
    # # Null case: 
    # ind_vals = linspace(1, nbins, len(kk[:,1])).astype(int)

    par_vals_averaged = array([median(velprofile_vals[:,2][ind_vals==ind]) for ind in range(1, nbins+1)])
    perp_vals_averaged = array([median(velprofile_vals[:,3][ind_vals==ind]) for ind in range(1, nbins+1)])

    if fill_nans:
        par_vals_averaged = interpolate_nans(par_vals_averaged)
        perp_vals_averaged = interpolate_nans(perp_vals_averaged)

    out = vstack([bin_vals[1:], par_vals_averaged, perp_vals_averaged]).T
    
    return out




    