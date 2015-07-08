import random
import os

import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, Max
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

structure = {
                'structure' : [625, 512, 128, 64],
                'activations' : 3 * [('sigmoid', 'relu')],
                'noise' : [GaussianNoise(0.01), None, None]
            }

def pretrain_deep_ae(params, X, tie_weights=True, batch_size=100, nb_epoch=5):
    '''
    A function for building and pretraining (interactively) a 
    deep autoencoder.

    Args:
        params (dict): A dictionary with the following fields:
                * `structure`: a list of ints that describe the
                        structure of the net, i.e., [10, 13, 2]
                * `activation`: a list of tuples of strings of 
                        length len(structure - 1) that describe the 
                        encoding and decoding activation function.
                        For example, [('sigmoid', 'relu'), ('sigmoid', 'relu')]
                * `noise` (optional): a list of keras layers or None that describe 
                        the noise you want to add. i.e., [GaussianNoise(0.01), None]
    '''
    if type(params) is not dict:
        raise TypeError('params must be of class `dict`.')
    for k in ['structure', 'activations']:
        if k not in params.keys():
            raise KeyError('key: `{}` must be in params dict'.format(k))

    if len(params['structure']) != (len(params['activations']) + 1):
        raise ValueError(
            'length of activations must be one less than length of structure.'
            )

    if 'noise' not in params.keys():
        params['noise'] = len(params['activations']) * [None]

    structure = params['structure']
    autoencoder = []
    for (inputs, hidden), (enc_act, dec_act), noise in zip(
            zip(
                structure, 
                structure[1:]
                ), 
            params['activations'], 
            params['noise']
        ):
    
        logger.info('Building {} x {} structure.'.format(inputs, hidden))
        autoencoder.append(Sequential())
        if noise is not None:
            encoder = containers.Sequential(
                          [
                              noise, 
                              Dense(inputs, hidden, activation=enc_act)
                          ]
                      )
        else:
            encoder = Dense(inputs, hidden, activation=enc_act)
        autoencoder[-1].add(
            AutoEncoder(
                    encoder=encoder,
                    decoder=Dense(hidden, inputs, activation=dec_act), 
                    output_reconstruction=False, 
                    tie_weights=tie_weights
                )
            )
        logger.info('Compiling...')
        autoencoder[-1].compile(loss='mse', optimizer=Adam())
        logger.info('Training...')
        try:
            autoencoder[-1].fit(X, X, batch_size=batch_size, nb_epoch=nb_epoch)
        except KeyboardInterrupt:
            logger.info('Training ended early...')
        X = autoencoder[-1].predict(X)
    return autoencoder

def unroll_deep_ae(structure, autoencoder, tie_weights=True):
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



