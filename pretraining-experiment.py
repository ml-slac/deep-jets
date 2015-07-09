import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers

from dl import pretrain_deep_ae, unroll_deep_ae
from utils import WeightedDataset


# -- swap with your own.
data = np.load('data-wprime-qcd.npy')

# -- load and process daa
X_ = np.array([x.ravel() for x in data['image']]).astype('float32')
y_ = data['signal'].astype('float32')

df = WeightedDataset(X_, y_)

buf = df.sample(500000)

n_train = 460000

X, y = buf[0][:n_train], buf[1][:n_train]
X_val, y_val = buf[0][n_train:], buf[1][n_train:]


# -- build pretrained nets.

params = {
            'structure' : [625, 512, 128, 64],
            'activations' : 3 * [('sigmoid', 'relu')],
            'noise' : [GaussianNoise(0.01), None, None],
            'optimizer' : Adam(),
            'loss' : ['mse', 'mse', 'mse']
         }

ae, config = pretrain_deep_ae(params, X[:10000])

model = unroll_deep_ae(ae, config)

model.compile(loss='mse', optimizer=Adam())

model.fit(X, X, batch_size=512)


clf = Sequential()

clf.add(model.layers[0].encoder)
clf.add(Dropout(0.1))
clf.add(Dense(64, 1))
clf.add(Activation('sigmoid'))
clf.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')

clf.fit(X, y, validation_data = (X_val, y_val), batch_size=100, nb_epoch=10, show_accuracy=True)



# -- test a big maxout net


mo = Sequential()
mo.add(MaxoutDense(625, 200))
mo.add(Dropout(0.4))
mo.add(MaxoutDense(200, 40))
mo.add(Dropout(0.4))
mo.add(Dense(40, 1))
mo.add(Activation('sigmoid'))

mo.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')
mo.fit(X, y, validation_data = (X_val, y_val), batch_size=100, nb_epoch=10, show_accuracy=True)





