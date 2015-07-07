'''
deepjets.py

an interface to Keras that allows physics users 
to manipulate and understand deep networks.
'''

import random
import os

import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, DenoisingAutoEncoder
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepNet(object):
	'''
	A wrapper class around Keras models -- allows activation inspection.
	'''
	def __init__(self, pretraining = True, model=Sequential()):
		super(DeepNet, self).__init__()
		self.model = model
		self.pretraining = pretraining

		self._is_pretrained = False

		self._is_compiled = []

		self.submodels = {}

		self.autoencoder = []

		self._model_cache = []

		self.weights = None

		self.callbacks = {}

	def add(layer, **kwargs):
		'''
		Functionality to add a Keras layer to a DeepNet
		'''
		loss, optimizer = None, None
		if 'loss' not in kwargs.keys():
			loss = 'mse'
		if 'optimizer' not in kwargs.keys():
			optimizer = Adam()
		logger.info('Adding {} layer'.format(layer.__class__))
		if type(layer) == AutoEncoder and not self.pretraining:
				raise TypeError(
					'pretraining = False set while trying to add an AutoEncoder'
					)
		else:
			# self._is_compiled.append(False)
			self._model_cache.append(Sequential())
			_model_cache[-1].add(layer)
			autoencoder[-1].compile(loss=loss, optimizer=optimizer)


	# def pretrain(X, **kwargs):
	# 	pass






		