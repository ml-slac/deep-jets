import numpy as np

from keras.layers import containers
from keras.models import Sequential
from keras.layers.convolutions import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers

from dl import pretrain_deep_ae, unroll_deep_ae
from utils import WeightedDataset


# -- Experiment mode
EXPERIMENT_MODE = False

if EXPERIMENT_MODE:
    for f in ['./viz/visualize.py', './dl/deepae.py', './utils/sampling.py', './utils/sampling.py']:
        %run f

# -- swap with your own.
data = np.load('data-wprime-qcd.npy')
data = data[data['jet_pt'] > 150]

# -- load and process daa
X_ = np.array([x.ravel() for x in data['image']]).astype('float32')
y_ = data['signal'].astype('float32')

df = WeightedDataset(X_, y_)

buf = df.sample(300000)

n_train = 260000

X, y = buf[0][:n_train], buf[1][:n_train]
X_val, y_val = buf[0][n_train:], buf[1][n_train:]

tau21 = data['tau_21'][df._ix_buf]
mass = data['jet_mass'][df._ix_buf]
pt = data['jet_pt'][df._ix_buf]

train_sample = df._ix_buf[:n_train]
test_sample = df._ix_buf[n_train:]

# -- build pretrained nets.

PRETRAINING = False

if PRETRAINING:
    params = {
                'structure' : [625, 256, 64, 28],
                'activations' : 2 * [('sigmoid', 'sigmoid')] + 1 * [('sigmoid', 'sigmoid')],
                'noise' : 4 * [Dropout(0.6)],
                'optimizer' : Adam(),
                'loss' : 2 * ['binary_crossentropy'] + 1 * ['binary_crossentropy']
             }

    ae, config = pretrain_deep_ae(params, X, nb_epoch=30, batch_size=512)

    model = unroll_deep_ae(ae, config)

    model.compile(loss='binary_crossentropy', optimizer=Adam())




    im = Sequential()
    im.add(Embedding(1, 625, W_constraint=NonNeg()))
    im.add(containers.Sequential(model.layers[0].encoder.layers[:-1]))

    w = model.layers[0].encoder.layers[-1].get_weights()

    im.add(Dense(64, 1, weights=[w[0][:, 0].reshape(64, 1), np.array(w[1][0]).reshape((1, ))]))

    im.add(Activation('sigmoid'))

    im.compile(loss='mse', optimizer=Adam())


    weights = model.layers[0].encoder.layers[0].get_weights()



    # model.fit(X, X, batch_size=512)


    clf = Sequential()

    clf.add(model.layers[0].encoder)
    clf.add(Dropout(0.1))
    clf.add(Dense(64, 1))
    clf.add(Activation('sigmoid'))
    clf.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')

    clf.fit(X, y, validation_data = (X_val, y_val), batch_size=100, nb_epoch=10, show_accuracy=True)

if not PRETRAINING:
    # -- test a big maxout net
    mo = Sequential()
    mo.add(MaxoutDense(625, 200, 5))
    mo.add(Dropout(0.3))
    mo.add(Dense(200, 64))
    mo.add(Activation('relu'))
    mo.add(Dropout(0.3))
    mo.add(Dense(64, 10))
    mo.add(Activation('relu'))
    mo.add(Dropout(0.3))
    mo.add(Dense(10, 1))
    mo.add(Activation('sigmoid'))

    mo.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')
    mo.fit(X, y, validation_data = (X_val, y_val), batch_size=100, nb_epoch=10, show_accuracy=True)

    # -- performance


    tau21 = data['tau_21']#[df._ix_buf]
    mass = data['jet_mass']#[df._ix_buf]
    pt = data['jet_pt']#[df._ix_buf]
    signal = data['signal'] == True
    background = data['signal'] == False

    weights = np.ones(data.shape[0])
    # -- going to match bkg to signal
    weights[signal] = get_weights(pt[signal], pt[background])

    discs = {}
    add_curve(r'\tau_{32}', 'red', calculate_roc(signal, tau21))
    fg = ROC_plotter(discs)
    fg.savefig('myroc.pdf')

    # -- nice viz

    hidden_act = Sequential()

    hidden_act.add(
            containers.Sequential(mo.layers[:-2])
        )
    hidden_act.compile('adam', 'mse')


    R = hidden_act.predict(X_, verbose=True)

    means = [data['jet_mass'][np.argsort(R[:, i])[::-1][:100]].mean() for i in xrange(20)]

    filter_grid(np.array([data['image'][np.argsort(R[:, i])[::-1][:100]].mean(axis=0) for i in xrange(20)])[np.argsort(means)], shape=(4, 5), cmap=custom_div_cmap(mincol='white', midcol='yellow', maxcol='red'), symmetric=False)


    top_n = 10

    title_map = {
        'jet_pt' : r'Jet $\overline{p_T} = %.2f \mathrm{GeV}$',
        'jet_mass' : r'Jet $\overline{ m } = %.2f \mathrm{GeV}$',
        'tau_21' : r'Jet $\overline{\tau_{21}} = %.2f$'
    }

    n_hidden = 20

    top_n_images = np.array(
        [
            data['image'][
                    np.argsort(R[:, i])[::-1][:top_n]
                ].mean(axis=0) 
            for i in xrange(n_hidden)
        ]
    )

    for feature_name, stylized in title_map.iteritems():
        feature = [data[feature_name][
                            np.argsort(R[:, i])[::-1][:top_n]
                        ].mean() 
                    for i in xrange(n_hidden)
                   ]
        plot = filter_grid(filters = top_n_images[np.argsort(feature)], 
                           labels = [r''+ stylized % v for v in np.sort(feature)], 
                           shape=(4, 5), 
                           cmap=cm.hot, 
                           symmetric=False)
        plot.savefig(feature_name + 'node_acts.pdf')
        


# filter_grid(np.array([data['image'][np.argsort(R[:, i])[::-1][:top_n]].mean(axis=0) for i in xrange(20)])[np.argsort(means)], labels=[r'Jet $\bar{p_T}=%.2f\mathrm{GeV}$' % m for m in np.sort(means)], )






