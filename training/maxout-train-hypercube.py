'''
maxout-train-hypercube.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to train a maxout net in a hypercube

'''
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation, Flatten, Merge 
from keras.layers.normalization import LRN2D
from keras.layers import containers
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold
from viz import *



class ROCModelCheckpoint(Callback):
    def __init__(self, filepath, X, y, weights, verbose=True):
        super(Callback, self).__init__()

        self.X, self.y, self.weights = X, y, weights

        self.verbose = verbose
        self.filepath = filepath
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        yh = self.model.predict(self.X, verbose=True).ravel()
        print yh[:30]
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


class NDWeights(object):
    """docstring for NDWeights"""
    def __init__(self, bins):
        super(NDWeights, self).__init__()
        self.bins = bins
    def fit(self, X, truth, reference):
        H_s, _ = np.histogramdd(X[truth == 1], bins=self.bins, normed=False)
        H_b, _ = np.histogramdd(X[truth == 0], bins=self.bins, normed=False)
        H_ref, _ = np.histogramdd(reference, bins=self.bins, normed=False)
        self.flat_cube_s = H_ref / H_s
        self.flat_cube_b = H_ref / H_b

    def predict(self, X, truth):
        ix = [(self.bins[i].searchsorted(X[:, i]) - 1) for i in xrange(len(self.bins))]
        ix = np.array(ix).T
        print ix
        weights = []
        for i, label in zip(ix, truth):
            if label == 1:
                w = np.copy(self.flat_cube_s[i[0]])
            else:
                w = np.copy(self.flat_cube_b[i[0]])
            for j in xrange(1, len(self.bins)):
                w = w[i[j]]
            weights.append(w)
        weights = np.array(weights)
        weights[np.isinf(weights)] = weights[np.isfinite(weights)].max()
        return weights

# X_train_image = np.load()

print 'Loading data...'
# data = np.load('../../jet-simulations/trainingset.npy')
data = np.load('/path/to/data.npy')
print 'there are {} training points'.format(data.shape[0])
print 'convering to images...'
X = data['image'].reshape((data.shape[0], 25 ** 2)).astype('float32')
y = data['signal'].astype('float32')


print 'extracting cube weights...'
signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

signal = (signal == 1)
background = (signal == False)



window = (tau_21 < 0.8) & (tau_21 > 0.2)
pt, mass, tau_21, signal, background = pt[window], mass[window], tau_21[window], signal[window], background[window]

n_obs = int(window.sum())

ref = np.zeros((n_obs, 3))
ref[:, 0] = pt
ref[:, 1] = mass
ref[:, 2] = tau_21

cube = np.zeros((n_obs / 2, 3))

cube[:, 0] = np.random.uniform(250, 300, n_obs / 2)
cube[:, 1] = np.random.uniform(65, 95, n_obs / 2)
cube[:, 2] = np.random.uniform(0.2, 0.8, n_obs / 2)


binning = (
            np.linspace(250, 300, 20),
            np.linspace(65, 95, 19),
            np.linspace(0.2, 0.8, 19)
          )

ndweights = NDWeights(binning)
ndweights.fit(ref, signal, cube)


cube_weights = ndweights.predict(ref, signal)
cube_weights[np.isinf(cube_weights)] = cube_weights[np.isfinite(cube_weights)].max()

print 'shuffling'

X, y = X[window], y[window]

idx = range(X.shape[0])
np.random.shuffle(idx)
X = X[idx][:1000000]
y = y[idx][:1000000]

cube_weights = cube_weights[idx].astype('float32')[:1000000]




try:
    kf = KFold(X.shape[0], 10)
    foldN = 1
    for train_ix, test_ix in kf:
        print 'Working on fold: {}'.format(foldN)

        print 'Building new submodel...'
        # -- build the model
        dl = Sequential()
        dl.add(MaxoutDense(256, 5, input_shape=(625, ), init='he_normal'))
        dl.add(Dropout(0.3))

        dl.add(MaxoutDense(128, 5, init='he_normal'))
        dl.add(Dropout(0.2))

        dl.add(Dense(64))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.2))

        dl.add(Dense(25))
        dl.add(Activation('relu'))
        dl.add(Dropout(0.3))

        dl.add(Dense(1))
        dl.add(Activation('sigmoid'))

        print 'compiling...'

        dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

        print 'training!'
        
    	h = dl.fit(X[train_ix], y[train_ix], batch_size=32, nb_epoch=50, show_accuracy=True, 
    	               validation_data=(X[test_ix], y[test_ix]), 
    	               callbacks = 
    	               [
    	                   EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
    	                   ModelCheckpoint('./trainings/final-slac-maxout-hypercube-unnormalized-logloss-cvFold{}.h5'.format(foldN), monitor='val_loss', verbose=True, save_best_only=True),
    	                   ROCModelCheckpoint('./trainings/final-slac-maxout-hypercube-unnormalized-roc-cvFold{}.h5'.format(foldN), X[test_ix], y[test_ix], cube_weights[test_ix], verbose=True)
    	               ],
                       sample_weight=cube_weights[train_ix]
                )
        foldN += 1
	               # sample_weight=np.power(weights, 0.7))
except KeyboardInterrupt:
	print 'ended early!'

# yhat = dl.predict(X, verbose=True).ravel()

# np.save('./yhat-cube.npy', yhat.astype('float32'))


with open('./trainings/final-slac-maxout-unnormalizedphypercube.yaml', 'wb') as f:
	f.write(dl.to_yaml())


