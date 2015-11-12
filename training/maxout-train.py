'''
maxout-train.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to train a maxout net

'''
import logging

from keras.models import Sequential, model_from_yaml
from keras.layers.core import *
from keras.layers import containers
from keras.optimizers import *
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
from sklearn.metrics import roc_curve, auc

from viz import *




LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)

class ROCModelCheckpoint(Callback):
    '''
    Callback to optimize AUC
    '''
    def __init__(self, filepath, X, y, weights, verbose=True):
        super(Callback, self).__init__()

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.filepath = filepath
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        yh = self.model.predict(self.X, verbose=True).ravel()
        log(yh[:30])
        fpr, tpr, _ = roc_curve(self.y, yh, sample_weight=self.weights)
        select = (tpr > 0.1) & (tpr < 0.9)
        current = auc(tpr[select], 1 / fpr[select])

        if current > self.best:
            if self.verbose > 0:
                print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                      % (epoch, 'AUC', self.best, current, self.filepath))
            self.best = current
            self.model.save_weights(self.filepath, overwrite=True)
        else:
            if self.verbose > 0:
                print("Epoch %05d: %s did not improve" % (epoch, 'AUC'))


# X_train_image = np.load()

log('Loading data...')
# data = np.load('../../jet-simulations/trainingset.npy')
data = np.load('/path/to/data.npy')
log('there are {} training points'.format(data.shape[0]))
log('convering to images...')
X = data['image'].reshape((data.shape[0], 25 ** 2)).astype('float32')
y = data['signal'].astype('float32')


log('extracting weights...')
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

signal = (signal == 1)
background = (signal == False)



# -- calculate the weights
weights = np.ones(data.shape[0])

# reference_distribution = np.random.uniform(250, 300, signal.sum())
reference_distribution = pt[background]

log('calculating weights...')
weights[signal] = get_weights(reference_distribution, pt[signal], 
    bins=np.linspace(250, 300, 200))

weights[background] = get_weights(reference_distribution, pt[background], 
    bins=np.linspace(250, 300, 200))
# weights[signal] = get_weights(pt[signal != 1], pt[signal], 
#   bins=np.concatenate((
#       np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
#   )

idx = range(X.shape[0])
np.random.shuffle(idx)
X = X[idx][:1000000]

# -- if you want to norm
# norms = np.sqrt((X ** 2).sum(-1))
# X = (X / norms[:, None])

y = y[idx][:1000000]
weights = weights[idx].astype('float32')[:1000000]



from sklearn.cross_validation import KFold
try:
    kf = KFold(X.shape[0], 10)
    foldN = 1
    for train_ix, test_ix in kf:
        log('Working on fold: {}'.format(foldN))

        log('Building new submodel...')
        # -- build the model
        dl = Sequential()
        dl.add(MaxoutDense(256, 5, input_shape=(625, ), init='he_uniform'))
        dl.add(Dropout(0.3))

        dl.add(MaxoutDense(128, 5, init='he_uniform'))
        dl.add(Dropout(0.2))

        dl.add(Dense(64))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.2))

        dl.add(Dense(25))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.3))

        dl.add(Dense(1))
        dl.add(Activation('sigmoid'))

        log('compiling...')

        dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

        log('training!')
        
    	h = dl.fit(X[train_ix], y[train_ix], batch_size=32, nb_epoch=50, show_accuracy=True, 
    	               validation_data=(X[test_ix], y[test_ix]), 
    	               callbacks = 
    	               [
    	                   EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
    	                   ModelCheckpoint('./SLACNormalized-final-logloss-cvFold{}.h5'.format(foldN), monitor='val_loss', verbose=True, save_best_only=True),
    	                   ROCModelCheckpoint('./SLACNormalized-final-roc-cvFold{}.h5'.format(foldN), X[test_ix], y[test_ix], weights[test_ix], verbose=True)
    	               ],
                       sample_weight=weights[train_ix]
                )
        foldN += 1
	               # sample_weight=np.power(weights, 0.7))
except KeyboardInterrupt:
	log('ended early!')


with open('./NewSLAC-final.yaml', 'wb') as f:
	f.write(dl.to_yaml())


