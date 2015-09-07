import numpy as np

from keras.layers import containers
from keras.models import Sequential
# from keras.layers.convolutions import Convolution2D, MaxPooling2D
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
data = data[data['jet_pt'] > 200]

dat = dat[(dat['jet_pt'] > 200) & (dat['jet_pt'] < 500)]
data = dat[np.abs(dat['jet_eta']) < 2]

# -- load and process daa
X_ = np.array([x.ravel() for x in data['image']]).astype('float32')
y_ = data['signal'].astype('float32')

df = WeightedDataset(X_, y_)

buf = df.sample(141417)

n_train = 115000

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
    maxout = Sequential()
    maxout.add(MaxoutDense(625, 200, 5))
    maxout.add(Dropout(0.3))
    maxout.add(Dense(200, 64))
    maxout.add(Activation('tanh'))
    maxout.add(Dropout(0.3))
    maxout.add(Dense(64, 16))
    maxout.add(Activation('tanh'))
    maxout.add(Dropout(0.3))
    maxout.add(Dense(16, 1))
    maxout.add(Activation('sigmoid'))

    maxout.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')
    mo.fit(X, y, validation_data = (X_val, y_val), batch_size=100, nb_epoch=10, show_accuracy=True)
    mo.fit(X, y, validation_split = 0.2, batch_size=100, nb_epoch=10, show_accuracy=True)

    y_dl = mo.predict(X_, verbose=True).ravel()
    # -- performance

    data = np.load('../correct_mass.npy')

    tau21 = data['tau_21']#[df._ix_buf]
    mass = data['jet_mass']#[df._ix_buf]
    pt = data['jet_pt']#[df._ix_buf]
    signal = data['signal'] == True
    background = data['signal'] == False

    weights = np.ones(data.shape[0])
    # -- going to match bkg to signal
    weights[background] = get_weights(pt[signal], pt[background])

    discs = {}
    add_curve(r'$\tau_{21}$', 'red', calculate_roc(signal, 2-tau21, weights=weights), discs)
    add_curve(r'Maxout Network', 'blue', calculate_roc(signal, y_dl, weights=weights), discs)
    add_curve(r'$m_{\mathrm{jet}}$', 'black', calculate_roc(signal, mass, weights=weights), discs)
    fg = ROC_plotter(discs, title=r'Tagging comparison -- match $b \longrightarrow s$.')
    fg.savefig('perfroc-bkg2sig.pdf')

    weights = np.ones(data.shape[0])
    # -- going to match bkg to signal
    weights[signal] = get_weights(pt[background], pt[signal])

    discs = {}
    add_curve(r'$\tau_{21}$', 'red', calculate_roc(signal, 2-tau21, weights=weights), discs)
    add_curve(r'Maxout Network', 'blue', calculate_roc(signal, y_dl, weights=weights), discs)
    add_curve(r'$m_{\mathrm{jet}}$', 'black', calculate_roc(signal, mass, weights=weights), discs)
    fg = ROC_plotter(discs, title=r'Tagging comparison -- match $s \longrightarrow b$.')
    fg.savefig('perfroc-sig2bkg.pdf')

    # -- nice viz

    hidden_act = Sequential()

    hidden_act.add(
            containers.Sequential(dl.layers[:-3])
        )
    hidden_act.compile('adam', 'mse')


    hidden_act_low = Sequential()

    hidden_act_low.add(
            containers.Sequential(dl.layers[:-6])
        )
    hidden_act_low.compile('adam', 'mse')


    R = hidden_act.predict(X_, verbose=True)

    means = [data['jet_mass'][np.argsort(R[:, i])[::-1][:10]].mean() for i in xrange(25)]
    means = [data['jet_mass'][np.argsort(R[:, i])[::-1][:10]].mean() for i in select]

    filter_grid(np.array([data['image'][np.argsort(R[:, i])[::-1][:3]].mean(axis=0) for i in select])[np.argsort(means)], shape=(5, 5), cmap=cm.hot, symmetric=False)


    [:200000]

    top_n = 25

    title_map = {
        'signal' : r'$\hat{p}_{\mathrm{signal}} = %.4f$',
        # 'jet_pt' : r'Jet $\overline{p_T} = %.2f \mathrm{GeV}$',
        'jet_mass' : r'Jet $\overline{ m } = %.2f \mathrm{GeV}$',
        'tau_21' : r'Jet $\overline{\tau_{21}} = %.2f$'
    }

    truth_map = {
        'Signal' : (data[:200000][data[:200000]['signal'] == 1], R[:200000][data[:200000]['signal'] == 1]),
        'Background' : (data[:200000][data[:200000]['signal'] == 0], R[:200000][data[:200000]['signal'] == 0]),
        'S and B' : (data[:200000], R[:200000])
    }

    n_hidden = 28


    p_hat_signal = [data['signal'][
                                np.argsort(R[:200000][:, i])[::-1][:111]
                            ].mean() 
                        for i in xrange(n_hidden)]

    for truth, (subset, hidden_repr) in truth_map.iteritems():
        top_n_images = np.array(
            [
                subset['image'][
                        np.argsort(hidden_repr[:, i])[::-1][:top_n]
                    ].mean(axis=0) 
                for i in xrange(n_hidden)
            ]
        )

        for feature_name, stylized in title_map.iteritems():
            feature = [subset[feature_name][
                                np.argsort(hidden_repr[:, i])[::-1][:top_n]
                            ].mean() 
                        for i in xrange(n_hidden)
                       ]
            # plot = filter_grid(filters = top_n_images,#[np.argsort(feature)], 
            plot = filter_grid(filters = top_n_images[np.argsort(p_hat_signal)], 
                               labels = [stylized % v for v in np.sort(feature)], 
                               shape=(5, 6), 
                               cmap=cm.hot, 
                               symmetric=False)
            plot.savefig(feature_name + '_' + truth.replace(' ', '_') + '_node_acts.pdf')
        


# filter_grid(np.array([data['image'][np.argsort(R[:, i])[::-1][:top_n]].mean(axis=0) for i in xrange(20)])[np.argsort(means)], labels=[r'Jet $\bar{p_T}=%.2f\mathrm{GeV}$' % m for m in np.sort(means)], )




    title_map = {
        # 'jet_pt' : r'Jet $\overline{p_T} = %.2f \mathrm{GeV}$',
        'jet_mass' : r'Jet $\overline{ m } = %.4f \mathrm{GeV}$',
        'tau_21' : r'Jet $\overline{\tau_{21}} = %.2f$'
    }

    truth_map = {
        'Signal' : (data[data['signal'] == 1], R[data['signal'] == 1]),
        'Background' : (data[data['signal'] == 0], R[data['signal'] == 0]),
        'S and B' : (data, R)
    }

    n_hidden = 20

    for feature_name, stylized in title_map.iteritems():
        # for truth, (subset, hidden_repr) in truth_map.iteritems():
        s_subset, s_hid = truth_map['Signal']
        b_subset, b_hid = truth_map['Background']
        s_top_n_images = np.array(
            [
                s_subset['image'][
                        np.argsort(s_hid[:, i])[::-1][:top_n]
                    ].mean(axis=0) 
                for i in xrange(n_hidden)
            ]
        )
        b_top_n_images = np.array(
            [
                b_subset['image'][
                        np.argsort(b_hid[:, i])[::-1][:top_n]
                    ].mean(axis=0) 
                for i in xrange(n_hidden)
            ]
        )
        feature = [s_subset[feature_name][
                            np.argsort(s_hid[:, i])[::-1][:top_n]
                        ].mean() 
                    for i in xrange(n_hidden)
                   ]
        plot = filter_grid(filters = top_n_images[np.argsort(feature)], 
                           labels = [stylized % v for v in np.sort(feature)], 
                           shape=(4, 5), 
                           cmap=cm.hot, 
                           symmetric=False)
        plot.savefig(feature_name + '_' + truth.replace(' ', '_') + '_node_acts.pdf')
        

plot = filter_grid(filters = diff[np.argsort(feature)] ** 2, 
                           # labels = [stylized % v for v in np.sort(feature)], 
                           shape=(4, 5), 
                           cmap=cm.hot, 
                           symmetric=False)
# filter_grid(np.array([data['image'][np.argsort(R[:, i])[::-1][:top_n]].mean(axis=0) for i in xrange(20)])[np.argsort(means)], labels=[r'Jet $\bar{p_T}=%.2f\mathrm{GeV}$' % m for m in np.sort(means)], )



reco = Sequential()
reco.add(MaxoutDense(16, 10, 5))
reco.add(Dropout(0.2))
reco.add(Dense(10, 5))
reco.add(Activation('tanh'))
reco.add(Dropout(0.2))
reco.add(Dense(5, 1))
reco.add(Activation('relu'))

reco.compile(loss='mse', optimizer=Adam())




fig = plt.figure(figsize=(15, 15), dpi=140)
ax = plt.subplot(111)
im = ax.hist2d(R[:, 1].ravel(), mass, bins = (100, 100), norm=LogNorm())
# ax.set_yscale('log')
# plt.colorbar()

fig = plt.figure(figsize=(15, 15), dpi=140)
ax = plt.subplot(111)
im = ax.hist2d(mass[(mass > 60) & (mass < 90)], np.power(10, reco_mass), bins = (100, 100), norm=LogNorm()); plt.show()

fig = plt.figure(figsize=(15, 15), dpi=140)
ax = plt.subplot(111)
plt.hist2d(mass[signal == False], reco_mass[signal == False], bins = (100, 100), norm=LogNorm())
plt.plot(*(2 * [np.linspace(0, 200, 200)]), ls='--', color='black')
plt.xlabel(r'Original $m$')
plt.ylabel(r'Reconstructed $\hat{m}$')
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.colorbar()
plt.show()

fig = plt.figure(figsize=(15, 15), dpi=140)
ax = plt.subplot(111)
plt.hist2d(mass[(mass > 75) & (mass < 100)], reco_mass[(mass > 75) & (mass < 100)], bins = (100, 100), norm=LogNorm())
plt.plot(*(2 * [np.linspace(75, 100, 200)]), ls='--', color='black')
plt.xlabel(r'Original $m$')
plt.ylabel(r'Reconstructed $\hat{m}$')
plt.xlim(75, 100)
plt.ylim(75, 100)
plt.colorbar()
plt.show()



est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='lad', n_jobs=4).fit(R, mass)

def activation_grid(hid, truth, labels=None, nfilters='all', shape=None):
    '''
    A tool for visualizing filters on a grid.

    Args:
        filters (iterable): each element should be an 
            image with len(image.shape) == 2

        nfilters: (str or int): out of the total filters, 
            how many to plot? If a str, must be 'all'

        shape (tuple): What shape of grid do we want?

        normalize (bool): do we normalize all filters to have 
            magnitude 1?

    Returns: 
        plt.figure
    '''
    
    NUMERICAL_NOISE_THRESH = 1e-3

    if nfilters == 'all':
        side_length = int(np.round(np.sqrt(hid.shape[1])))
    else:
        side_length = int(np.round(np.sqrt(nfilters)))

    fig = plt.figure(figsize=(15, 15), dpi=140)

    if shape is None:
        grid_layout = gridspec.GridSpec(side_length, side_length)
        nplots = side_length ** 2
    else:
        grid_layout = gridspec.GridSpec(shape[0], shape[1])
        nplots = shape[0] * shape[1]
        # GmtoT1osfCpLCw6lzpnXh79y
    plt.title('plots')
    grid_layout.update(wspace=1.0, hspace=1.0) # set the spacing between axes. 

    for i in range(hid.shape[1]):
        ax = plt.subplot(grid_layout[i])
    
        plt.hist(hid[truth, i].ravel(), log=False, bins = np.linspace(-1, 1, 35), histtype='step', color='red')
        plt.hist(hid[truth == False, i].ravel(), log=False, bins = np.linspace(-1, 1, 35), histtype='step', color='blue')

        if i % 10 == 0:
            logger.info('{} of {} completed.'.format(i, nplots))
        plt.axis('off')
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        if labels is not None:
            plt.title(labels[i])
        plt.subplots_adjust(hspace = 1, wspace=1)

    return fig

plt.hist(R[signal, 0].ravel(), log=False, bins = np.linspace(0, 3, 35), histtype='step', color='red')
plt.hist(R[background, 0].ravel(), log=False, bins = np.linspace(0, 3, 35), histtype='step', color='blue')





def corr_grid(hid, mass, labels=None, nfilters='all', shape=None):
    '''
    A tool for visualizing filters on a grid.

    Args:
        filters (iterable): each element should be an 
            image with len(image.shape) == 2

        nfilters: (str or int): out of the total filters, 
            how many to plot? If a str, must be 'all'

        shape (tuple): What shape of grid do we want?

        normalize (bool): do we normalize all filters to have 
            magnitude 1?

    Returns: 
        plt.figure
    '''
    
    NUMERICAL_NOISE_THRESH = 1e-3

    if nfilters == 'all':
        side_length = int(np.round(np.sqrt(hid.shape[1])))
    else:
        side_length = int(np.round(np.sqrt(nfilters)))

    fig = plt.figure(figsize=(35, 27), dpi=140)

    if shape is None:
        grid_layout = gridspec.GridSpec(side_length, side_length)
        nplots = side_length ** 2
    else:
        grid_layout = gridspec.GridSpec(shape[0], shape[1])
        nplots = shape[0] * shape[1]
        # GmtoT1osfCpLCw6lzpnXh79y
    plt.title('plots')
    grid_layout.update(wspace=1.0, hspace=1.0) # set the spacing between axes. 

    for i in range(hid.shape[1]):
        ax = plt.subplot(grid_layout[i])
        
        plt.hist2d(hid[:, i].ravel(), mass, bins=(100, 100), norm=LogNorm())
        # plt.hist(hid[truth, i].ravel(), log=True, bins = np.linspace(0, 3, 35), histtype='step', color='red')
        # plt.hist(hid[truth == False, i].ravel(), log=True, bins = np.linspace(0, 3, 35), histtype='step', color='blue')

        if i % 10 == 0:
            logger.info('{} of {} completed.'.format(i, nplots))
        # plt.axis('off')
        # ax.yaxis.set_visible(False)
        # ax.xaxis.set_visible(False)
        plt.xlabel('activation')
        plt.ylabel('Jet Mass')
        if labels is not None:
            plt.title(labels[i])
        # plt.subplots_adjust(hspace = 0, wspace=0)

    return fig










# -- build the model
dl = Sequential()
dl.add(Dense(625, 500))
dl.add(Activation('relu'))

dl.add(Dropout(0.2))
dl.add(Dense(500, 256))
dl.add(Activation('relu'))

dl.add(Dropout(0.2))
dl.add(Dense(256, 128))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(128, 64))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(64, 28))
dl.add(Activation('relu'))

dl.add(Dropout(0.1))
dl.add(Dense(28, 1))
dl.add(Activation('sigmoid'))

dl.compile(loss='binary_crossentropy', optimizer=Adam(), class_mode='binary')

# -- train!
h = maxout.fit(X_train, y_train, batch_size=512, nb_epoch=20, show_accuracy=True, 
               validation_data=(X_test, y_test), callbacks=[EarlyStopping(verbose=True, patience=2)])




from sklearn.preprocessing import StandardScaler

mass_reconstruction = Sequential()

mass_reconstruction.add(MaxoutDense(25, 30, 7, W_regularizer=l2(0.0001)))
mass_reconstruction.add(Dense(30, 15, activation='relu', W_regularizer=l2(0.0001)))
mass_reconstruction.add(Dense(15, 5, activation='relu', W_regularizer=l2(0.0001)))
mass_reconstruction.add(Dense(5, 1, W_regularizer=l2(0.0001)))

scaler = StandardScaler()

m_scaled = scaler.fit_transform(mass)


mass_reconstruction.compile('adam', 'mse')

mass_reconstruction.fit(R[:n_train], m_scaled[:n_train], 
                        validation_data=(R[n_train:], m_scaled[n_train:]), 
                        batch_size=200, 
                        nb_epoch=100
                        )


m_reconstructed = scaler.inverse_transform(mass_reconstruction.predict(R, verbose=True))




plt.hist(mass[(signal == 0) & (y_dl < 0.2)], color='blue', bins=150, histtype='step', label='QCD with $\hat{p} < 0.2$', normed=True)
plt.hist(mass[(signal == 0) & (y_dl < 0.5)], color='green', bins=150, histtype='step', label='QCD with $\hat{p} < 0.5$', normed=True)
plt.hist(mass[(signal == 0) & (y_dl > 0.5)], color='orange', bins=150, histtype='step', label='QCD with $\hat{p} \geq 0.5$', normed=True)
plt.hist(mass[(signal == 0) & (y_dl > 0.8)], color='red', bins=150, histtype='step', label='QCD with $\hat{p} \geq 0.8$', normed=True)
plt.title('Jet Mass distribution for QCD as a function of $\hat{p}$')
plt.xlabel('Jet Mass (GeV)')
plt.ylim((0, 0.025))
plt.legend()


def normalize_rows(x):
    def norm1d(a):
        return a / a.sum()
    x = np.array([norm1d(r) for r in x])
    return x




plt.imshow(np.flipud(H) / np.array([signal_mass.tolist() for _ in xrange(10)]), extent=(c.min(), c.max(), 0, 1), aspect='auto', interpolation='none')
plt.imshow(np.flipud(H), extent=(c.min(), c.max(), 0, 1), aspect='auto', interpolation='none')
plt.xlabel('Jet Mass (GeV)')
plt.ylabel(r'$\hat{y}$')
plt.title(r'PDF of QCD Jet Mass, binned in $\hat{y}$')
cb = plt.colorbar()
cb.set_label(r'$P(\mathrm{mass} \vert \hat{y})$')



plt.hist2d(mass[(mass > 85) & (mass < 95)], m_reconstructed[(mass > 85) & (mass < 95)], bins=200)
plt.xlabel('Original Mass (GeV)')
plt.ylabel('Reconstructed Mass (GeV)')
plt.ylim(80, 100)
plt.xlim(85, 95)
plt.colorbar()











