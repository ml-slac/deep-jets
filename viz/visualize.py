'''
visualize.py
author: Luke de Oliveira (lukedeo@stanford.edu)

Utilities and functions to inspect neural net filters.
'''
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap 
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

def filter_grid(filters, nfilters='all', shape=None, normalize=True):
    '''
    A tool for visualizing filters on a grid.

    Args:
        filters (iterable): each element should be an 
            image with len(image.shape) == 2


    '''
    
    NUMERICAL_NOISE_THRESH = 1e-4

    if nfilters == 'all':
        side_length = int(np.round(np.sqrt(len(filters))))
    else:
        side_length = int(np.round(np.sqrt(nfilters)))

    cma = custom_div_cmap(50)
    fig = plt.figure(figsize=(15, 15), dpi=140)

    if shape is None:
        grid_layout = gridspec.GridSpec(side_length, side_length)
        nplots = side_length ** 2
    else:
        grid_layout = gridspec.GridSpec(shape[0], shape[1])
        nplots = shape[0] * shape[1]

    grid_layout.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

    for i, filt in enumerate(filters):
        ax = plt.subplot(grid_layout[i])
        if normalize:
            filt /= np.sum(filt ** 2)

        # -- trim off absurd values.
        abs_max = np.percentile(np.abs(filt), 95)

        # -- trim out numerical zero noise
        filt[np.abs(filt) < NUMERICAL_NOISE_THRESH] = 0.0

        image = ax.imshow(filt, interpolation='nearest', 
                cmap=cma, vmin=-abs_max, vmax=abs_max)
        if i % 10 == 0:
            logger.info('{} of {} completed.'.format(i, nplots))
        plt.axis('off')
        plt.subplots_adjust(hspace = 0, wspace=0)

    return fig




class LayerProxy(object):
    """docstring for LayerProxy"""
    def __init__(self, L):
        super(LayerProxy, self).__init__()
        self.W, self.b = L.get_weights()
        if type(L) == Maxout
        _, self.inputs, self.outputs = self.W.shape

        
        




def factors(n):    
    return set(reduce(list.__add__, 
            ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))




def get_filter(net, layer=0, node=0, thresh=1e-8):
    # if layer is 0:
        # return net.architecture[0].W[node, :]
    # filt = np.eye(net[0].W.shape[1])
    filt = np.eye(net[0].W[:,1].shape[0])
    ctr = 0
    for L in net:
        W = L.W.T
        filt = np.asmatrix(W) * np.asmatrix(filt)
        if ctr is layer:
            break
        ctr += 1
    filt = filt[node, :] #/ np.sum(filt[node, :])
    filt = np.array(filt.tolist()[0])
    filt[np.abs(filt) < thresh] = 0
    return filt




rec = get_filter([dae.encoder, ae2.encoder, ae3.encoder], 2, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));
plt.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)



fig = plt.figure(figsize=(15, 15), dpi=140)

gs1 = gridspec.GridSpec(14, 14)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 



for i in xrange(0, 14 ** 2):
    ax = plt.subplot(gs1[i])
    rec = get_filter([dae.encoder, ae2.encoder], 1, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));

    ima = ax.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)

    # plt.title(r''+title)
    sys.stdout.write('\r{} of {} completed.'.format(i, 14**2))
    sys.stdout.flush()
    plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
fig.text(0.3, 0.91, r'Jet Image filters from denoising autoencoders, layer 2')

fig.savefig('jetfilters-layer2.pdf')












fig = plt.figure(figsize=(15, 15), dpi=140)

gs1 = gridspec.GridSpec(9, 9)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 


side_length = 9

for i in xrange(0, side_length ** 2):
    ax = plt.subplot(gs1[i])
    rec = get_filter([dae.encoder, ae2.encoder, ae3.encoder], 2, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));

    ima = ax.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)

    # plt.title(r''+title)
    sys.stdout.write('\r{} of {} completed.'.format(i, side_length ** 2))
    sys.stdout.flush()
    plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
fig.text(0.3, 0.91, r'Jet Image filters from denoising autoencoders')

fig.savefig('jetfilters-layer3.pdf')




fig = plt.figure(figsize=(15, 15), dpi=140)

gs1 = gridspec.GridSpec(5, 5)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 



for i in xrange(0, 5 ** 2):
    ax = plt.subplot(gs1[i])
    rec = get_filter([dae.encoder, ae2.encoder, ae3.encoder, ae4.encoder], 3, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));

    ima = ax.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)

    # plt.title(r''+title)
    sys.stdout.write('\r{} of {} completed.'.format(i, 5**2))
    sys.stdout.flush()
    plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
fig.text(0.3, 0.91, r'Jet Image filters from denoising autoencoders, layer 4')

fig.savefig('jetfilters-layer4.pdf')










fig = plt.figure(figsize=(15, 15), dpi=140)

gs1 = gridspec.GridSpec(3, 3)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 



for i in xrange(0, 3 ** 2):
    ax = plt.subplot(gs1[i])
    rec = get_filter([dae.encoder, ae2.encoder, ae3.encoder, ae4.encoder, ae5.encoder], 4, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));

    ima = ax.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)

    # plt.title(r''+title)
    sys.stdout.write('\r{} of {} completed.'.format(i, 3**2))
    sys.stdout.flush()
    plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
fig.text(0.3, 0.91, r'Jet Image filters from denoising autoencoders, layer 5')

fig.savefig('jetfilters-layer5.pdf')











fig = plt.figure(figsize=(15, 15), dpi=140)

gs1 = gridspec.GridSpec(1, 2)
gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 



for i in xrange(0, 2):
    ax = plt.subplot(gs1[i])
    rec = get_filter([dae.encoder, ae2.encoder, ae3.encoder, ae4.encoder, ae5.encoder, ae6.encoder], 5, i).reshape(25, 25); rec /= np.sqrt(np.sum(rec ** 2)); _ma = np.max(np.abs(rec));

    ima = ax.imshow(rec, interpolation='nearest', cmap=cma, vmin=-_ma, vmax=_ma)

    # plt.title(r''+title)
    sys.stdout.write('\r{} of {} completed.'.format(i, 2))
    sys.stdout.flush()
    plt.axis('off')
    plt.subplots_adjust(hspace = 0, wspace=0)
fig.text(0.3, 0.91, r'Jet Image filters from denoising autoencoders, layer 6')

fig.savefig('jetfilters-layer6.pdf')



