'''
maxout-test.py

author: Luke de Oliveira (lukedeo@stanford.edu)

description: script to load a maxout net

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
        help='path to DNN weights', required=True)

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
		from keras.optimizers import *
		from keras import regularizers
		from keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError:
    	sys.stderr.write('[ERROR] Keras not found!')
    	sys.exit(1)

    log('Construction Maxout architecture')

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
		# -- normalize the jet images
		data = (data / np.sqrt((data ** 2).sum(-1))[:, None])
	else:
		log('No normalization needed.')

	log('writing predictions.')
	# -- run this test data through the conv net.
	yhat = dl.predict(data, verbose=True).ravel()

	SAVE_FILE = args.output
	log('Saving DNN outputs to {}'.format(SAVE_FILE))
	np.save(SAVE_FILE, yhat.astype('float32'))
	log('Done')

