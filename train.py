import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, MaxoutDense, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.colors import LinearSegmentedColormap 
# from matplotlib.colors import LogNorm

# class DrawWeights(keras.callbacks.Callback):

#     def __init__(self, figsize, layer_id=0, param_id=0, weight_slice=(slice(None), 0)):
#         self.layer_id = layer_id
#         self.param_id = param_id
#         self.weight_slice = weight_slice
#         # Initialize the figure and axis
#         self.fig = plt.figure(figsize=figsize)
#         self.ax = self.fig.add_subplot(1, 1, 1)

#     def on_train_begin(self):
#         self.imgs = []

#     def on_batch_end(self, batch, indices, loss, accuracy):
#         # Get a snapshot of the weight matrix every 5 batches
#         if batch % 5 == 0:
#             # Access the full weight matrix
#             weights = self.model.layers[self.layer_id].params[self.param_id].get_value()
#             # Create the frame and add it to the animation
#             img = self.ax.imshow(weights[self.weight_slice], interpolation='nearest')
#             self.imgs.append(img)

#     def on_train_end(self):
#         # Once the training has ended, display the animation
#         anim = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False)
#         plt.show()

train = np.load('./wprime800_QCD200-600_train.npz')
test = np.load('./wprime800_QCD200-600_test.npz')
weights = np.load('./wprime800_QCD200-600_train_weights.npz')['weights']


# -- build the model
dl = Sequential()
dl.add(Dense(625, 500, W_regularizer=l2(0.0001)))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(500, 256, W_regularizer=l2(0.0001)))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(256, 128, W_regularizer=l2(0.0001)))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(128, 64, W_regularizer=l2(0.0001)))
dl.add(Activation('tanh'))

dl.add(Dropout(0.1))
dl.add(Dense(64, 25))
dl.add(Activation('tanh'))

dl.add(Dropout(0.1))
dl.add(Dense(25, 1))
dl.add(Activation('sigmoid'))

dl.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')

# -- train!
dl.fit(train['X'], train['y'], validation_data=(test['X'], test['y']), 
						 batch_size=256, 
						 nb_epoch=100, 
						 callbacks=[
						 			EarlyStopping(verbose=True, patience=2)
						 			], 
						 sample_weight=weights, 
						 show_accuracy=True, 
						 verbose=2)


with open('deepjets.yaml') as f:
	f.write(dl.to_yaml())

dl.save_weights('deepjets.h5')







