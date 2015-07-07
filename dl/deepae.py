import random
import os

import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_dae(structure, X, tie_weights=True, denoising = -1):
    autoencoder = []
    for inputs, hidden in zip(structure, structure[1:]):
        logger.info('Building {} x {} structure.'.format(inputs, hidden))
        autoencoder.append(Sequential())
        if denoising > 0:
            encoder = containers.Sequential(
                          [
                              GaussianNoise(denoising), 
                              Dense(inputs, hidden, activation='sigmoid')
                          ]
                      )
        else:
            encoder = Dense(inputs, hidden, activation='sigmoid')
        autoencoder[-1].add(
            AutoEncoder(
                    encoder=encoder,
                    decoder=Dense(hidden, inputs, activation='relu'), 
                    output_reconstruction=False, 
                    tie_weights=tie_weights
                )
            )
        logger.info('Compiling...')
        autoencoder[-1].compile(loss='mse', optimizer=Adam())
        logger.info('Training...')
        try:
            autoencoder[-1].fit(X, X, batch_size=100, nb_epoch=5)
        except KeyboardInterrupt:
            logger.info('Training ended early...')
        X = autoencoder[-1].predict(X)
    return autoencoder

def unroll_dae(structure, autoencoder, tie_weights=True):
    encoder = []
    decoder = []
    for layer_nb, (inputs, hidden) in enumerate(zip(structure, structure[1:])):
        logger.info('Unpacking structure from level {}.'.format(layer_nb))
        encoder.append(Dense(inputs, hidden, activation='sigmoid'))
        encoder[-1].set_weights(autoencoder[layer_nb].get_weights()[:2])
        decoder.insert(0, Dense(hidden, inputs, activation='relu'))
        decoder[0].set_weights(autoencoder[layer_nb].get_weights()[2:])

    encoder_sequence = containers.Sequential(encoder)
    decoder_sequence = containers.Sequential(decoder)

    stacked_autoencoder = Sequential()

    stacked_autoencoder.add(AutoEncoder(encoder=encoder_sequence, 
                                        decoder=decoder_sequence, 
                                        output_reconstruction=False, 
                                        tie_weights=True))
    return stacked_autoencoder