'''
A set of helper functions for plotting scientific data

https://github.com/williamgilpin/swimpy

'''

from numpy import *
from scipy import *
# from matplotlib.pyplot import *
from matplotlib.colors import LinearSegmentedColormap
from pylab import *
from cycler import cycler

rcParams['font.family'] = 'Open Sans'
rcParams['font.weight'] = 'bold'
# # These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
#tableau20 = [(0.59375, 0.257813, 0.886719),(0.800781, 0.0625, 0.460938), (0.278431, 0.788235, 0.478431), (0.917647, 0.682353, 0.105882), (0.372549, 0.596078, 1.), (0.8, 0.8, .8), (1.0, .3882, .2784)];

tableaugray = [(123,102,210),(207,207,207),(165,172,175),(143,135,130),(96,99,106),(65,68,81)]

tab_purp = [(255,192,218),(216,152,186),(180,177,155),(153,86,136),(139,124,110),\
         (220,95,189),(95,90,65),(171,106,213),(219,212,197)]

tab_og = [(255,127,15), (172,217,141),(60,183,204),(50,162,81),(255,217,74),(184,90,13),\
(152,217,228),(134,180,169),(57,115,124),(204,201,77),(130,133,59)]

tab_br = [(255,182,176),(240,39,32),(181,200,226),(172,97,60),(44,105,176),(233,195,155),\
(221,201,180),(181,223,253),(172,135,99),(107,163,214),(244,115,122),(189,10,54)]

tab_extra = [(208,152,238), (166,153,232), (219,212,197)]

all_colors = tableau20 + tableaugray + tab_purp + tab_og + tab_br + tab_extra

for i in range(len(all_colors)):  
    r, g, b = all_colors[i]  
    all_colors[i] = (r, g, b)
    all_colors[i] = (r / 255., g / 255., b / 255.)  

bcolor1 = (53 / 255., 56 / 255., 57 / 255.)
bcolor2 = (59 / 255., 60 / 255., 54 / 255.)

# rcParams['axes.color_cycle'] = all_colors     # DEPRECATED
rcParams['axes.prop_cycle'] = cycler('color', all_colors)

rcParams['axes.labelcolor'] = bcolor1
rcParams['axes.edgecolor'] = bcolor2
rcParams['xtick.color'] = bcolor2
rcParams['ytick.color'] = bcolor2

rcParams['legend.frameon'] = False

rcParams['axes.facecolor'] = 'none'

rcParams['lines.linewidth'] = 2

rcParams['figure.facecolor'] = 'none'
rcParams['savefig.facecolor'] = 'none'

rcParams['image.cmap'] = 'gray'
rcParams['image.interpolation'] = 'nearest'

rcParams['figure.figsize'] = (6, 6)

def cmap1D(col1, col2, N):
    '''Generate a continuous colormap between two values
    
    Parameters
    ----------
    
    col1 : tuple of ints
        RGB values of final color
        
    col2 : tuple of ints
        RGB values of final color
    
    N : int
        The number of values to interpolate
        
    Returns
    -------
    
    col_list : list of tuples
        An ordered list of colors for the colormap
    
    '''
    
    col1 = array([item/255. for item in col1])
    col2 = array([item/255. for item in col2])
    
    vr = list()
    for ii in range(3):
        vr.append(linspace(col1[ii],col2[ii],N))
    colist = array(vr).T
    return [tuple(thing) for thing in colist]


def fixed_aspect_ratio(ratio):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    xvals,yvals = gca().axes.get_xlim(),gca().axes.get_ylim()
    
    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    gca().set_aspect(ratio*(xrange/yrange), adjustable='box')



def print_status(iter_val, final_val, max_val=79):
    '''
    Print a progress bar in a loop calculation
    
    Parameters
    ----------
    
    iter_val : int
        The current value of the iteration variable
    
    final_val : int
        The maximum value that the iteration variable 
        will eventually attain
    
    max_val : int
        The maximum width of a line. Defaults to PEP recommendation of 79
    
    
    Note: This only works in Python 3 because of the unique properties of 
            the print() function.
    '''
 
    iter_val_scaled = int((iter_val/final_val)*max_val)
    remainder = max_val-int(((final_val-1)/final_val)*max_val)
    
    print('', end="\r")
    print('[', end="")
    for jj in range(iter_val_scaled+remainder):
        print ('#', end="")
    for jj in range(max_val-iter_val_scaled-remainder):     
        print (' ', end="")
    print(']',end="")


def cmap1D(col1, col2, N):
    '''Generate a continuous colormap between two values
    
    Parameters
    ----------
    
    col1 : tuple of ints
        RGB values of final color
        
    col2 : tuple of ints
        RGB values of final color
    
    N : int
    
        The number of values to interpolate
        
    Returns
    -------
    
    col_list : list of tuples
        An ordered list of colors for the colormap
    
    '''
    
    col1 = array([item/255. for item in col1])
    col2 = array([item/255. for item in col2])
    
    vr = list()
    for ii in range(3):
        vr.append(linspace(col1[ii],col2[ii],N))
    colist = array(vr).T
    return [tuple(thing) for thing in colist]


def plot_colormap(arr,color1=(90,112,255),color2=(235,16,118),**kwargs):
    '''Plot a series of traces stored in an array 
    using a progressive colormap
    
    Parameters
    ----------
    
    array : array
        Array containing lines to plot
    
    color1 : 3-tuple
        RGB values (0-255) for the starting color
        
    color2 : 3-tuple
        RGB values (0-255) for the ending color
        
    kwargs : 
        additional keyword arguments get passed to 
        matplotlib's plot function. Use the "markers" option
        to change the marker style
    
    '''
    
    n_colors = len(arr)
    
    all_colors = cmap1D(color1, color2, n_colors)
    color_ind=0
    
    figure()
    hold(True)
    for row in arr:
        plot(row,color=all_colors[color_ind],**kwargs)
        color_ind = color_ind + 1

# This slows everything down because of equation rendering. It also breaks fonts
# rcParams['text.usetex'] = True

#rcParams['figure.savefig.bbox']= 'none'

# for ii in range(50):
#     plot([rand(), rand()],[rand(), rand()])
#savefig("out.pdf")