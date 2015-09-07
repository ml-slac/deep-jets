import logging

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, MaxoutDense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import PReLU

import numpy as np




# X = np.load('./JI_X_window.npy').astype('float32')
# Xgs = np.load('./JI_gs_window.npy').astype('float32')
# Xconv = np.load('./JI_conv_window.npy').astype('float32')
# # X = np.load('./JI_X_image.npy').astype('float32')
# y = np.load('./JI_y_window.npy').astype('float32')
# weights = np.load('./JI_weights_window.npy').astype('float32')



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info('loading data...')

with np.load('./JI_train.npz') as d:
    train = {'X' : d['X'], 'y' : d['y'], 'weights' : d['weights']}
with np.load('./JI_test.npz') as d:
    test = {'X' : d['X'], 'y' : d['y']}
# n_train = int(0.75 * X.shape[0])

# X_train = X[:n_train]
# X_train_gs = Xgs[:n_train]
# X_train_conv = Xconv[:n_train]
# Y_train = y[:n_train]

# X_test = X[n_train:]
# X_test_gs = Xgs[n_train:]
# X_test_conv = Xconv[n_train:]
# Y_test = y[n_train:]

logger.info('building model...')


# -- build the model
# raw = Sequential()
# raw.add(MaxoutDense(625, 512, 10))

# raw.add(Dropout(0.5))
# raw.add(Dense(512, 256))
# # raw.add(Activation('tanh'))
# raw.add(PReLU((256, )))

# raw.add(Dropout(0.3))
# raw.add(Dense(256, 128))
# # raw.add(Activation('tanh'))
# raw.add(PReLU((128, )))

# raw.add(Dropout(0.2))
# raw.add(Dense(128, 64))
# raw.add(Activation('tanh'))

# gaussian = Sequential()

# gaussian.add(Dense(625, 700))
# gaussian.add(Activation('relu'))
# gaussian.add(Dropout(0.1))

# gaussian.add(Dense(700, 300))
# gaussian.add(Activation('relu'))

# gaussian.add(Dropout(0.3))
# gaussian.add(Dense(300, 64))
# gaussian.add(Activation('relu'))


# convolved = Sequential()

# convolved.add(Dense(625, 700))
# convolved.add(Activation('relu'))
# convolved.add(Dropout(0.1))

# convolved.add(Dense(700, 300))
# convolved.add(Activation('relu'))

# convolved.add(Dropout(0.3))
# convolved.add(Dense(300, 64))
# convolved.add(Activation('relu'))

# dl = Sequential()
# # dl.add(Merge([raw, gaussian, convolved], mode='concat'))
# dl.add(Dense(64 * 3, 25))
# dl.add(Activation('tanh'))

# dl.add(Dense(25, 1))
# dl.add(Activation('sigmoid'))

# -- build the model
dl = Sequential()
# dl.add(Merge([raw, gaussian], mode='concat'))
# dl.add(Dense(1000, 512))
dl.add(MaxoutDense(625, 512, 10))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(512, 256, 6))
# dl.add(Activation('tanh'))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(256, 64, 6))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(64, 25, 10))

dl.add(Dropout(0.1))
dl.add(Dense(25, 1))
dl.add(Activation('sigmoid'))

# dl.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')



logger.info('compiling model...')
dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

NAME = './SLACNetWindow-mass[65-95]-pt[250-300]800GeV'

logger.info('starting training...')
try:
    h = dl.fit(train['X'], train['y'], batch_size=256, nb_epoch=20, show_accuracy=True, 
                   validation_data=(test['X'], test['y']), 
                   callbacks = [
                       EarlyStopping(verbose=True, patience=6),
                       ModelCheckpoint(NAME + '-weights.h5', monitor='val_loss', verbose=True, save_best_only=True)
                   ], 
                   sample_weight=train['weights'])
except KeyboardInterrupt:
    logger.info('Ended early..')


with open(NAME + '.yaml', 'w') as f:
    f.write(dl.to_yaml())







# #-- START building conv net...

# # Initial arguments
# nr_kernels1 = 10 # arbitrary choice
# kernel_size = 25  # arbitrary choice
# nr_channels = 1  # RGB inputs => 3 channels

# logger.info('building model...')

# conv = Sequential()

# conv.add(Convolution2D(nr_kernels1,
#                         nr_channels,
#                         kernel_size,
#                         kernel_size,
#                         border_mode='full'))

# conv.add(Activation('relu'))
# conv.add(MaxPooling2D(poolsize=(2, 2))) # arbitrary choice
# conv.add(Dropout(0.25))

# #7
# conv.add(Flatten())


# #8
# # The dimensions of each image is (N = (X.shape[2]) ** 2),
# # We had two downsampling layers of 2x2 maxpooling, so we divide each dimension twice by 2 (/2 /2).
# # The input to this layer is the 64 "channels" that the previous layer outputs. Thus we have a layer of
# # nr_kernels * (N / 2 / 2) * (N / 2 / 2)

# # flat_layer_size = nr_kernels2 * (X.shape[2] / 2) ** 2
# flat_layer_size = 3136
# # flat_layer_size = nr_kernels2 * (X.shape[2] / 2) ** 2
# final_layer_size=256 # I chose it
# conv.add(Dense(flat_layer_size, final_layer_size))
# conv.add(Activation('relu'))
# conv.add(Dropout(0.3))

# conv.add(MaxoutDense(final_layer_size, 64, 5))

# conv.add(Dropout(0.3))
# #9
# conv.add(Dense(64, 1))
# conv.add(Activation('sigmoid'))
# logger.info('compiling conv...')

# # let's train the conv using SGD + momentum (how original).
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# conv.compile(loss='binary_crossentropy', optimizer=sgd, class_mode='binary')

# logger.info('starting training...')


# conv.fit(X_train, Y_train, batch_size=512, 
#     nb_epoch=300, show_accuracy=True, 
#     validation_data = (X_test, Y_test), 
#     callbacks=[
#             ModelCheckpoint('./SLAC-convnet-weights-long.h5', monitor='val_loss', verbose=True, save_best_only=True), 
#             EarlyStopping(patience=7, verbose=True)
#         ]
#     )


# with open('./SLAC-convnet.yaml', 'w') as f:
#     f.write(conv.to_yaml())


