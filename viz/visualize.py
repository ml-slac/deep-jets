'''
visualize.py
author: Luke de Oliveira (lukedeo@stanford.edu)

Utilities and functions to inspect neural net filters.
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap 
from matplotlib.colors import LogNorm
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def custom_div_cmap(numcolors=21, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    '''
    Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 21 colors.  

    Colors can be specified in any way understandable 
    by matplotlib.colors.ColorConverter.to_rgb()
    '''
    cmap = LinearSegmentedColormap.from_list(
                    name=name, 
                    colors=[mincol, midcol, maxcol], 
                    N=numcolors
                )
    return cmap

def filter_grid(filters, labels=None, nfilters='all', shape=None, normalize=True, cmap=None, symmetric=True):
    '''
    A tool for visualizing filters on a grid.

    Args:
        filters (iterable): each element should be an 
            image with len(image.shape) == 2

        nfilters: (str or int): out of the total filters, 
            how many to plot? If a str, must be 'all'

        shape (tuple): What shape of grid do we want?

        normalize (bool): do we normalize all filters to have 
            magnitude 1?

    Returns: 
        plt.figure
    '''
    
    NUMERICAL_NOISE_THRESH = 1e-3

    if nfilters == 'all':
        side_length = int(np.round(np.sqrt(len(filters))))
    else:
        side_length = int(np.round(np.sqrt(nfilters)))

    if cmap is None:
        cma = custom_div_cmap(50)
    else:
        cma = cmap
    fig = plt.figure(figsize=(15, 15), dpi=140)

    if shape is None:
        grid_layout = gridspec.GridSpec(side_length, side_length)
        nplots = side_length ** 2
    else:
        grid_layout = gridspec.GridSpec(shape[0], shape[1])
        nplots = shape[0] * shape[1]
        # GmtoT1osfCpLCw6lzpnXh79y
    plt.title('plots')
    grid_layout.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

    for i, filt in enumerate(filters):
        ax = plt.subplot(grid_layout[i])
        if normalize:
            filt /= np.sum(filt ** 2)

        # -- trim off absurd values.
        # abs_max = np.percentile(np.abs(filt), 98)
        abs_max = np.max(np.abs(filt))

        # -- trim out numerical zero noise
        filt[np.abs(filt) < NUMERICAL_NOISE_THRESH] = 0.0
        if symmetric:
            image = ax.imshow(filt, interpolation='nearest', 
                    cmap=cma, vmin=-abs_max, vmax=abs_max)
        else:
            image = plt.imshow(filt, interpolation='nearest', cmap=cma)
        if i % 10 == 0:
            logger.info('{} of {} completed.'.format(i, nplots))
        plt.axis('off')
        if labels is not None:
            plt.title(labels[i])
        plt.subplots_adjust(hspace = 0, wspace=0)

    return fig

class FilterInspectionLayer(object):
    
    def __init__(self, L):
        super(FilterInspectionLayer, self).__init__()
        self.W, self.b = L.get_weights()
        if self.W.shape == 3:
            _, self.inputs, self.outputs = self.W.shape
        else:
            self.inputs, self.outputs = self.W.shape