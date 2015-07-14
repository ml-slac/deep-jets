import random
import os

import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense, ActivityRegularization
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# params = {
#             'structure' : [625, 512, 128, 64],
#             'activations' : 3 * [('sigmoid', 'relu')],
#             'noise' : [GaussianNoise(0.01), None, None],
#             'optimizer' : Adam(),
#             'loss' : ['mse', 'mse', 'mse']
#          }

def pretrain_deep_ae(params, X, tie_weights=True, batch_size=100, nb_epoch=5, validation_data=None):
    '''
    A function for building and greedily pretraining (interactively) 
    a deep autoencoder.

    Args:
        params (dict): A dictionary with the following fields:
                * `structure`: a list of ints that describe the
                        structure of the net, i.e., [10, 13, 2]
                * `activations`: a list of tuples of strings of 
                        length len(structure - 1) that describe the 
                        encoding and decoding activation function.
                        For example, [('sigmoid', 'relu'), ('sigmoid', 'relu')]
                * `noise` (optional): a list of keras layers or None that describe 
                        the noise you want to add. i.e., [GaussianNoise(0.01), None]
                * `optimizer` (optional): one of the keras optimizers
                * `loss` (optional): a list of the loss functions to use.

        X (numpy.ndarray): the data to perform the unsupervised pretraining on.

        tie_weights (bool): tied or untied autoencoders.

        batch_size and nb_epoch: should be self explanatory...

    Usage:

        >>> params = {
        ....'structure' : [625, 512, 128, 64],
        ....'activations' : 3 * [('sigmoid', 'relu')],
        ....'noise' : [GaussianNoise(0.01), None, None],
        ....'optimizer' : Adam(),
        ....'loss' : ['mse', 'mse', 'mse']
        ....}
        >>> ae, p = pretrain_deep_ae(params, X)

    Returns:
        a tuple (list, params), where list is a list of keras.Sequential().
    '''
    # -- check for logic errors.
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
        logger.info('noise specifications not specified -- default to None')
        params['noise'] = len(params['activations']) * [None]


    if 'optimizer' not in params.keys():
        logger.info('optimization specifications not specified -- using Adam()')
        params['optimizer'] = Adam()

    if 'loss' not in params.keys():
        logger.info('loss specifications not specified -- using MSE')
        params['optimizer'] = len(params['activations']) * ['mse']

    structure = params['structure']
    autoencoder = []

    # -- loop through the parameters
    for (inputs, hidden), (enc_act, dec_act), noise, loss in zip(
            zip(
                structure,    # -- number of inputs
                structure[1:] # -- number of outputs
                ), 
            params['activations'], 
            params['noise'],
            params['loss']
        ):
    
        logger.info('Building {} x {} structure.'.format(inputs, hidden))
        autoencoder.append(Sequential())
        if noise is not None:
            # -- noise should be a keras layer, so it can be in a Sequential() 
            logger.info('using noise of type {}'.format(type(noise)))
            encoder = containers.Sequential(
                          [
                              noise, 
                              Dense(inputs, hidden, activation=enc_act), 
                              ActivityRegularization(l1=0.001)
                          ]
                      )
        else:
            # -- just a regular (non-denoising) ae.
            encoder = containers.Sequential(
                          [
                              Dense(inputs, hidden, activation=enc_act), 
                              ActivityRegularization(l1=0.001)
                          ]
                      )
            # encoder = Dense(inputs, hidden, activation=enc_act)

        # -- each element of the list is a Sequential(), so we add.   
        autoencoder[-1].add(
            AutoEncoder(
                    encoder=encoder,
                    decoder=Dense(hidden, inputs, activation=dec_act), 
                    output_reconstruction=False, 
                    tie_weights=tie_weights
                )
            )
        logger.info('Compiling...')
        # -- each layer has it's own loss, but there is a global optimizer.
        logger.info('Loss: {}, Optimizer: {}'.format(loss, type(params['optimizer'])))
        autoencoder[-1].compile(loss=loss, optimizer=params['optimizer'])
        logger.info('Training...')

        # -- we allow people to end the training of each unit early.
        try:
            autoencoder[-1].fit(X, X, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=validation_data)
        except KeyboardInterrupt:
            logger.info('Training ended early...')

        # -- embed in the new code space.
        X = autoencoder[-1].predict(X)

    return autoencoder, params

def unroll_deep_ae(autoencoder, params, tie_weights=True):
    '''
    Takes an autoencoder list generated by `pretrain_deep_ae` and
    unrolls it to make a deep autoencoder. NOTE this doesn't
    compile anything! This is simply a wrapper around the 
    unrolling process to make it easier.

    Args:
        autoencoder (list): a list of keras layers.
        params (dict): the param dict returned by `pretrain_deep_ae` 
        tie_weights (bool): whether or not to make the weights tied.
    
    Usage:

        >>> params = {
        ....'structure' : [625, 512, 128, 64],
        ....'activations' : 3 * [('sigmoid', 'relu')],
        ....'noise' : [GaussianNoise(0.01), None, None],
        ....'optimizer' : Adam(),
        ....'loss' : ['mse', 'mse', 'mse']
        ....}
        >>> model = unroll_deep_ae(*pretrain_deep_ae(params, X))

    Returns:
        keras.Sequential: a keras sequential model with one layer
            which is the unrolled autoencoder.
    '''
    encoder = []
    decoder = []

    structure = params['structure']

    for (layer_nb, (inputs, hidden)), (enc_act, dec_act) in zip(
            enumerate(
                    zip(
                        structure, 
                        structure[1:]
                    )
                ), 
            params['activations']
        ):

        logger.info('Unpacking structure from level {}.'.format(layer_nb))
        encoder.append(Dense(inputs, hidden, activation=enc_act))
        encoder[-1].set_weights(autoencoder[layer_nb].get_weights()[:2])
        decoder.insert(0, Dense(hidden, inputs, activation=dec_act))
        decoder[0].set_weights(autoencoder[layer_nb].get_weights()[2:])

    encoder_sequence = containers.Sequential(encoder)
    decoder_sequence = containers.Sequential(decoder)

    stacked_autoencoder = Sequential()

    stacked_autoencoder.add(AutoEncoder(encoder=encoder_sequence, 
                                        decoder=decoder_sequence, 
                                        output_reconstruction=False, 
                                        tie_weights=tie_weights))
    return stacked_autoencoder






