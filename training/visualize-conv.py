from scipy.ndimage import convolve

from keras.layers import containers
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense, Activation, Merge
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# %run ../viz/visualize.py
# %run ../viz/performance.py
from viz import *
from likelihood import *




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
    	filt = filt.copy()
        ax = plt.subplot(grid_layout[i])
        if normalize:
            filt /= np.s
            um(filt ** 2)

        # -- trim off absurd values.
        # abs_max = np.percentile(np.abs(filt), 98)
        abs_max = np.max(np.abs(filt))

        # -- trim out numerical zero noise
        # filt[np.abs(filt) < NUMERICAL_NOISE_THRESH] = 0.0
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



PLOT_DIR = './plots/arxiv/%s'

data = np.load('../FINAL_SAMPLE.npy')

print '{} jets before preselection'.format(data.shape[0])

signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

import deepdish.io as io

net = io.load('./SLACNetConv-final-logloss.h5')

import matplotlib.cm as cm

fg = filter_grid(net['layer_0']['param_0'].reshape(64, 11, 11), normalize=False, cmap=cm.YlGnBu, symmetric=False)

fg.savefig(PLOT_DIR % 'conv-filts.pdf')


signal = (signal == 1)
background = (signal == False)

# -- calculate the weights
weights = np.ones(data.shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

weights[signal] = get_weights(reference_distribution, pt[signal], 
	bins=np.linspace(250, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
	bins=np.linspace(250, 300, 200))
# weights[signal] = get_weights(pt[signal != 1], pt[signal], 
# 	bins=np.concatenate((
# 		np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
# 	)



sig_jets = data['image'][signal == True]
bkg_jets = data['image'][signal == False]

sig_mean = np.average(sig_jets, axis=0)#, weights=weights[signal == True])
bkg_mean = np.average(bkg_jets, axis=0)#, weights=weights[signal == False])

sig_mean_ben = np.average(ben['image'][ben['signal'] == 1], axis=0)
bkg_mean_ben = np.average(ben['image'][ben['signal'] == 0], axis=0)


def _filt_diff(s, b, w, border='constant'):
	return convolve(s, w, mode=border, cval=0.0) - convolve(b, w, mode=border, cval=0.0)


fg = filter_grid([_filt_diff(sig_mean, bkg_mean, np.sign(w) * np.sqrt(np.abs(w))) for w in net['layer_0']['param_0'].reshape(64, 11, 11)], normalize=False, symmetric=True)

fg.savefig(PLOT_DIR % 'conv-diffs-global.pdf')






