'''
convnet-test.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to load a CNN

'''

import argparse
import logging
import sys
import numpy as np


LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str,
        help='path to CNN weights', required=True)

    parser.add_argument('--normed', action='store_true',
        help='are the weights expecting a normed jet image?', required=True)

    parser.add_argument('--data', type=str, required=True,
        help='path to data file (.npy)')

    parser.add_argument('--output', type=str, required=True,
        help='path to write output predictions.')

    args = parser.parse_args()

    try:
    	from keras.models import Sequential, model_from_yaml
		from keras.layers.core import *
		from keras.layers.convolutional import MaxPooling2D, Convolution2D
		from keras.optimizers import *
		from keras import regularizers
		from keras.callbacks import EarlyStopping, ModelCheckpoint
		from keras.layers.normalization import LRN2D
    except ImportError:
    	sys.stderr.write('[ERROR] Keras not found!')
    	sys.exit(1)

    log('Construction CNN architecture')
	FILTER_SIZES = [(11, 11), (3, 3), (3, 3), (3, 3)]

	dl = Sequential()
	dl.add(Convolution2D(32, *FILTER_SIZES[0], input_shape=(1, 25, 25), border_mode='full', W_regularizer=regularizers.l2(0.01)))
	dl.add(Activation('relu'))
	dl.add(MaxPooling2D((2, 2)))

	dl.add(Convolution2D(32, *FILTER_SIZES[1], border_mode='full', W_regularizer=regularizers.l2(0.01)))
	dl.add(Activation('relu'))
	dl.add(MaxPooling2D((3, 3)))

	dl.add(Convolution2D(32, *FILTER_SIZES[2], border_mode='full', W_regularizer=regularizers.l2(0.01)))
	dl.add(Activation('relu'))
	dl.add(MaxPooling2D((3, 3)))

	dl.add(LRN2D())
	dl.add(Flatten())

	dl.add(Dropout(0.2))

	dl.add(Dense(64))
	dl.add(Activation('relu'))
	dl.add(Dropout(0.1))

	dl.add(Dense(1))
	dl.add(Activation('sigmoid'))


	WEIGHT_FILE = args.weights
	log('Loading weights from {}'.format(WEIGHT_FILE))

	dl.load_weights(WEIGHT_FILE)

	log('Compiling')
	dl.compile(loss='binary_crossentropy', optimizer='sgd', class_mode='binary')


	TEST_DATA = args.data

	log('Loading data from {}'.format(TEST_DATA))
	data = np.load(TEST_DATA)['image']
	data = data.reshape(data.shape[0], -1)

	if args.normalize:
		log('Normalization needed. Normalizing...')
		# -- normalize and reshape the jet images
		data = (data / np.sqrt((data ** 2).sum(-1))[:, None]).reshape((data.shape[0], 1, 25, 25))
	else:
		log('No normalization needed.')

	log('writing predictions.')
	# -- run this test data through the conv net.
	yhat = dl.predict(data, verbose=True).ravel()

	SAVE_FILE = args.output
	log('Saving CNN outputs to {}'.format(SAVE_FILE))
	np.save(SAVE_FILE, yhat.astype('float32'))
	log('Done')

