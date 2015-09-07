from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense, Activation, Merge
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

%run ./viz/visualize.py
%run ./viz/performance.py
%run ./dl/deepae.py
%run ./utils/sampling.py
%run ./likelihood.py

PLOT_DIR = './plots/8-25/%s'


# dat = np.load('../deep-jets/data/processed/wprime-800-qcd-8-17.npy')
dat = np.load('../jet-simulations/slac-data.npy')
print '{} jets before preselection'.format(dat.shape[0])


# -- apply preselection
dat = dat[(dat['jet_pt'] > 250) & (dat['jet_pt'] < 300)]
# dat = dat[dat['jet_mass'] < 500]
data = dat[(dat['jet_mass'] > 65) & (dat['jet_mass'] < 95)]
# data = dat[np.abs(dat['jet_eta']) < 2]

signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']
pt_selection = (pt > 20)
selection = pt_selection
# -- mass window selection
# mass_cut = (dat['jet_mass'] < 95) & (dat['jet_mass'] > 65)
# data = dat[mass_cut]

print '{} jets after preselection'.format(data.shape[0])
_ = data['signal']
signal_pct = _.mean()
print '{}% signal'.format(signal_pct)


idx = range(data.shape[0])
np.random.shuffle(idx)
data = data[idx]


X_ = np.array([x.ravel() for x in data['image']]).astype('float32')
# X_ = X_ * data['total_energy'][:, np.newaxis]
y_ = data['signal'].astype('float32')



n_train = int(0.7 * data.shape[0])

print 'Using {} training points, and {} testing points.'.format(n_train, data.shape[0] - n_train)

X_train, y_train, X_test, y_test = X_[:n_train], y_[:n_train], X_[n_train:], y_[n_train:]

# X_gaussian_train, X_gaussian_test = X_gaussian[:n_train], X_gaussian[n_train:]

# X_image = data['image'].astype('float32').reshape((data.shape[0], 1, 25, 25))
# # X_2channel = X_2channel[idx]
# # X_2channel = np.empty((data.shape[0], 2, 25, 25))
# X_gs = np.empty((data.shape[0], 25 * 25))

# for i in xrange(data.shape[0]):
# 	if i % 100:
# 		print 'Jet {} of {}'.format(i + 1, data.shape[0])
# 	X_gs[i] = ndimage.filters.gaussian_filter(X_image[i][0], 0.8, mode='constant', cval=0.0, truncate=2).ravel()


# X_conv = np.empty((data.shape[0], 25 * 25))

# for i in xrange(data.shape[0]):
# 	if i % 100:
# 		print 'Jet {} of {}'.format(i + 1, data.shape[0])
# 	X_conv[i] = ndimage.convolve(X_image[i][0], k, mode='constant', cval=0.0).ravel()







# im[max((m - r), 0) : min((m + r), im.shape[0]), max((n - r), 0) : min((n + r), im.shape[0])]


# P_mass = Likelihood1D(np.concatenate((np.array([0, 25]), np.linspace(25, 35, 7), np.linspace(35, 160, 100), np.array([160, 900]))))




signal, pt, mass, tau_21,  = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

# -- plot some kinematics...
plt.hist(pt[signal == True], bins=np.linspace(200, 1000, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$")
plt.hist(pt[signal == False], bins=np.linspace(200, 1000, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$p_T$ (GeV)')
plt.ylabel('Count')
plt.title(r'Unweighted Jet $p_T$ distribution')
plt.legend()
plt.savefig(PLOT_DIR % 'unweighted-pt-distribution.pdf')
plt.show()


# -- calculate the weights
weights = np.ones(data.shape[0])
weights[signal == 1] = get_weights(pt[signal != 1], pt[signal == 1], 
	bins=np.linspace(200, 400, 150))
# weights[signal == 1] = get_weights(pt[signal != 1], pt[signal == 1], 
# 	bins=np.concatenate((
# 		np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
# 	)



# -- plot reweighted...
plt.hist(pt[signal == True], bins=np.linspace(200, 1000, 500), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist(pt[signal == False], bins=np.linspace(200, 1000, 500), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$p_T$ (GeV)')
plt.ylabel('Count')
plt.title(r'Weighted Jet $p_T$ distribution (matched to QCD)')
plt.legend()
         
plt.savefig(PLOT_DIR % 'weighted-pt-distribution.pdf')
plt.show()



# -- plot weighted mass
plt.hist(mass[signal == True], bins=np.linspace(0, 400, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist(mass[signal == False], bins=np.linspace(0, 400, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$m$ (GeV)')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m$ distribution ($p_T$ matched to QCD)')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-mass-distribution.pdf')
plt.show()



# -- plot weighted tau_21
plt.hist(tau_21[signal == True], bins=np.linspace(0, 0.95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist(tau_21[signal == False], bins=np.linspace(0, 0.95, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$\tau_{21}$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $\tau_{21}$ distribution ($p_T$ matched to QCD)')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-tau21-distribution.pdf')
plt.show()

P_mass = Likelihood1D(np.concatenate((np.array([0, 25]), np.linspace(25, 35, 5), np.linspace(35, 160, 20), np.array([160, 900]))))
P_mass.fit(mass[:n_train][signal[:n_train] == 1], mass[:n_train][signal[:n_train] == 0], weights=(weights[:n_train][signal[:n_train] == 1], weights[:n_train][signal[:n_train] == 0]))
mass_likelihood = P_mass.predict(mass)

# -- plot weighted mass likelihood
plt.hist((mass_likelihood[signal == True]), bins=np.linspace(0, 10, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist((mass_likelihood[signal == False]), bins=np.linspace(0, 10, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m$ likelihood distribution ($p_T$ matched to QCD)')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-mass-likelihood-distribution.pdf')
plt.show()




mass_bins = np.linspace(64.99, 95.01, 20)#, np.linspace(35, 159.99, 30), np.array([160, 900])))
tau_bins = np.concatenate((np.array([0, 0.01]), np.linspace(0.01001, 0.8199999999, 20), np.array([0.82, 1])))
P_2d = Likelihood2D(mass_bins, tau_bins)
P_2d.fit((mass[:n_train][signal[:n_train] == 1], tau_21[:n_train][signal[:n_train] == 1]), (mass[:n_train][signal[:n_train] == 0], tau_21[:n_train][signal[:n_train] == 0]), weights=(weights[:n_train][signal[:n_train] == 1], weights[:n_train][signal[:n_train] == 0]))
P_2d.fit((mass[signal == 1], tau_21[signal == 1]), (mass[signal == 0], tau_21[signal == 0]), weights=(weights[signal == 1], weights[signal == 0]))
mass_nsj_likelihood = P_2d.predict((mass, tau_21))
# -- plot weighted mass + nsj likelihood
plt.hist((mass_nsj_likelihood[signal == True]), bins=np.linspace(0, 100, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist((mass_nsj_likelihood[signal == False]), bins=np.linspace(0, 100, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m, \tau_{21}$ likelihood distribution ($p_T$ matched to QCD)')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-mass-nsj-likelihood-distribution.pdf')
plt.show()

pt_selection = (pt > 250) & (pt < 300)
mass_selection = (mass > 65) & (mass < 95)
selection = pt_selection & mass_selection# & (np.log(mass_nsj_likelihood) > 2)
# pt_selection = (pt > 300) & (pt < 500)
# pt_selection = (pt > 500)
# pt_selection = (pt > 20)
# [pt_selection]

signal, pt, mass, tau_21, y_dl = signal[selection], pt[selection], mass[selection], tau_21[selection], y_dl[selection]
signal, pt, mass, tau_21, y_dl = signal[idx], pt[idx], mass[idx], tau_21[idx], y_dl[idx]




# -- calculate the weights
weights = np.ones(signal.shape[0])
weights[signal == 1] = get_weights(pt[signal != 1], pt[signal == 1], 
	bins=np.linspace(249.999, 300.001, 150))
# weights[signal == 1] = get_weights(pt[signal != 1], pt[signal == 1], 
# 	bins=np.concatenate((
# 		np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
# 	)

discs = {}
add_curve(r'$\tau_{21}$', 'red', 
          calculate_roc(signal, 2 - tau_21, weights=weights), discs)
# add_curve(r'$\tau_{21}$', 'black', 
          # calculate_roc(signal[selection], 2 - tau_21_old[selection], weights=weights[selection]), discs)
# add_curve(r'$m_{\mathrm{jet}}$ (1D likelihood)', 'black', calculate_roc(signal[n_train:][selection[n_train:]], np.log(mass_likelihood)[n_train:][selection[n_train:]], weights=weights[n_train:][selection[n_train:]], bins=1000000), discs)
add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood)', 'blue', calculate_roc(signal, (mass_nsj_likelihood), bins=1000000, weights=weights), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD Tagging comparison -- match $s \longrightarrow b$." + '\n' + r'Jet $p_T\in[200, 350]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$GeV')
fg.savefig(PLOT_DIR % 'mass-likelihood-nsj-roc-500-1000.pdf')

plt.show()

# --------
# -- training

raw = Sequential()
raw.add(Dense(625, 500))
raw.add(Activation('relu'))

gaussian = Sequential()
gaussian.add(Dense(625, 500))
gaussian.add(Activation('relu'))

# -- build the model
dl = Sequential()
# dl.add(Merge([raw, gaussian], mode='concat'))
# dl.add(Dense(1000, 512))
dl.add(MaxoutDense(625, 512, 10))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(512, 256, 6))
# dl.add(Activation('tanh'))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(256, 64, 6))

dl.add(Dropout(0.1))
dl.add(MaxoutDense(64, 25, 10))

dl.add(Dropout(0.1))
dl.add(Dense(25, 1))
dl.add(Activation('sigmoid'))

dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')


h = dl.fit(X_train, y_train, batch_size=512, nb_epoch=20, show_accuracy=True, 
               validation_data=(X_test, y_test), 
               callbacks = [
                   EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
                   ModelCheckpoint('./BIGDATASLACNet-weights.h5', monitor='val_loss', verbose=True, save_best_only=True)
               ], 
               sample_weight=weights[:n_train])

y_dl = dl.predict(X_, verbose=True).ravel()

from sklearn.lda import LDA
lda = LDA()
lda.fit(X_train, y_train)
# lda.fit(X_[selection], y_[selection])
yld = lda.predict_proba(X_)
yld = yld[:, 1]


DNN_kin = Likelihood2D(np.linspace(-4, 6.2, 6), np.linspace(0, 1, 50))
DNN_kin.fit((np.log(mass_nsj_likelihood + 1e-6)[signal == 1], y_dl[signal == 1]), (np.log(mass_nsj_likelihood + 1e-6)[signal == 0], y_dl[signal == 0]), weights=(weights[signal == 1], weights[signal == 0]))
likelihood2 = DNN_kin.predict((np.log(mass_nsj_likelihood + 1e-6), y_dl))

add_curve(r'Deep Net', 'orange', calculate_roc(signal, y_dl, weights=weights, bins=1000000), discs)
add_curve(r'Deep Net + $(m_{\mathrm{jet}}, \tau_{21})$', 'black', calculate_roc(signal, likelihood2, weights=weights, bins=1000000), discs)
add_curve(r'FLD', 'green', calculate_roc(signal[selection], yld[selection], weights=weights[selection], bins=1000000), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD Tagging comparison -- match $s \longrightarrow b$." + 
	'\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$')
fg.savefig(PLOT_DIR % 'dl-roc.pdf')

# -- small windows..

mass_window = (mass > mass_min) & (mass < mass_max)
discs = {}
add_curve(r'$\tau_{21}$', 'red', 
          calculate_roc(signal[mass_window], 2 - tau_21[mass_window], weights=weights[mass_window]), discs)
add_curve(r'$m_{\mathrm{jet}}$ (likelihood)', 'black', calculate_roc(signal[mass_window], mass_likelihood[mass_window], weights=weights[mass_window]), discs)
add_curve(r'Deep Net', 'blue', calculate_roc(signal[mass_window], y_dl[mass_window], weights=weights[mass_window]), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD Tagging comparison -- match $s \longrightarrow b$." + 
	'\n' + r'Jet $p_T\in[250, 300]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in [65, 95]$ GeV')
fg.savefig(PLOT_DIR % 'dl-roc-masswindow.pdf')


# -- plot nsj
plt.hist((tau_21[(signal == True) & mass_window]), bins=np.linspace(0, 1, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[(signal == 1) & mass_window])
plt.hist((tau_21[(signal == False) & mass_window]), bins=np.linspace(0, 1, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
plt.ylabel('Count')
plt.title(r'Weighted $\tau_{21}$ distribution ($p_T$ matched to QCD)' + '\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-nsj-masswindow-distribution.pdf')
plt.show()


mass_nsj_window = (mass > mass_min) & (mass < mass_max) & (tau_21 < 0.4) 
discs = {}
add_curve(r'$\tau_{21}$', 'red', 
          calculate_roc(signal[mass_nsj_window], 2 - tau_21[mass_nsj_window], weights=weights[mass_nsj_window]), discs)
add_curve(r'$m_{\mathrm{jet}}$ (likelihood)', 'black', calculate_roc(signal[mass_nsj_window], mass_likelihood[mass_nsj_window], weights=weights[mass_nsj_window]), discs)
add_curve(r'Deep Net', 'blue', calculate_roc(signal[mass_nsj_window], y_dl[mass_nsj_window], weights=weights[mass_nsj_window]), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD Tagging comparison -- match $s \longrightarrow b$." + 
	'\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV, $\tau_{21}$ < 0.4')
fg.savefig(PLOT_DIR % 'dl-roc-massnsjwindow.pdf')




# -- 



mass_min, mass_max = [65, 110]

plt.hist2d(mass[(signal == 1) & (mass > mass_min) & (mass < mass_max)], y_dl[(signal == 1) & (mass > mass_min) & (mass < mass_max)], bins=50)
plt.ylabel(r'Deep Net output ($\hat{y}$)')
plt.xlabel('Jet mass (GeV)')
plt.xlim(mass_min, mass_max)
# plt.xlim(0, 1)
plt.colorbar()
plt.title(r"$W' \rightarrow WZ$ Jet mass vs. $\hat{y}$, " + '\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV')
plt.savefig(PLOT_DIR % 'dl-output-mass-hist-signal.pdf')
plt.show()


plt.hist2d(mass[(signal == 0)  & (mass > mass_min) & (mass < mass_max)], y_dl[(signal == 0)  & (mass > mass_min) & (mass < mass_max)], bins=50)
plt.ylabel(r'Deep Net output ($\hat{y}$)')
plt.xlabel('Jet mass (GeV)')
plt.xlim(mass_min, mass_max)
# plt.ylim()
# plt.xlim(0, 1)
plt.colorbar()
plt.title(r"QCD Jet mass vs. $\hat{y}$, " + '\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV')
plt.savefig(PLOT_DIR % 'dl-output-mass-hist-bkg.pdf')
plt.show()

# -- mass likelihood

plt.hist2d(mass_likelihood[(signal == 1) & (mass > mass_min) & (mass < mass_max)], y_dl[(signal == 1) & (mass > mass_min) & (mass < mass_max)], bins=20)
plt.ylabel(r'Deep Net output ($\hat{y}$)')
plt.xlabel('Jet mass likelihood')
# plt.xlim(mass_min, mass_max)
# plt.xlim(0, 1)
plt.colorbar()
plt.title(r"$W' \rightarrow WZ$ Jet mass vs. $\hat{y}$, " + '\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV')
plt.savefig(PLOT_DIR % 'dl-output-mass-likelihood-signal.pdf')
plt.show()


plt.hist2d(mass_likelihood[(signal == 0)  & (mass > mass_min) & (mass < mass_max)], y_dl[(signal == 0)  & (mass > mass_min) & (mass < mass_max)], bins=20)
plt.ylabel(r'Deep Net output ($\hat{y}$)')
plt.xlabel('Jet mass likelihood')
# plt.xlim(mass_min, mass_max)
# plt.ylim()
# plt.xlim(0, 1)
plt.colorbar()
plt.title(r"QCD Jet mass vs. $\hat{y}$, " + '\n' + r'Jet $p_T\in[200, 1000]$ $\mathrm{GeV},\vert\eta\vert<2$, $m_{\mathrm{jet}}\in (65, 110)$ GeV')
plt.savefig(PLOT_DIR % 'dl-output-mass-likelihood-bkg.pdf')
plt.show()




# -- hidden activations

hidden_act = Sequential()

hidden_act.add(
        containers.Sequential(dl.layers[:-3])
    )
hidden_act.compile('sgd', 'mse')


R = hidden_act.predict(X_, verbose=True)



def normalize_rows(x):
    def norm1d(a):
        return a / a.sum()
    x = np.array([norm1d(r) for r in x])
    return x

H, b_x, b_y = np.histogram2d(
	mass[selection][(signal[selection] == 0)], 
	y_dl[selection][(signal[selection] == 0)], 
	bins=(np.linspace(65, 95, 35), np.linspace(0, 1, 35)), 
	normed=True)




plt.imshow(np.flipud(normalize_rows(H.T)), extent=(65, 95, 0, 1), aspect='auto', interpolation='nearest')
plt.xlabel('QCD Jet Mass (GeV)')
plt.ylabel(r'$\hat{y}$ (Deep Net output)')
plt.title(r'PDF of QCD Jet Mass, binned vs. Deep Net output' + '\n' + 
	r'Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [65, 95]$ GeV')
cb = plt.colorbar()
cb.set_label(r'$P(\mathrm{mass} \vert \hat{y})$')
plt.savefig(PLOT_DIR % 'mass-dist-yhat-unweighted.pdf')
plt.show()


pdfs = np.flipud(normalize_rows(H.T))

signal_mass_dist, _ = np.histogram(mass[(signal == 1) & (mass < 130)], bins = b_x, norm)



H, b_x, b_y = np.histogram2d(
	mass[(signal == 0) & (mass < 130)], 
	y_dl[(signal == 0) & (mass < 130)], 
	bins=(100, 20), 
	normed=True)

pdfs = np.flipud(normalize_rows(H.T))

signal_mass_dist, _ = np.histogram(mass[(signal == 1) & (mass < 130)], bins = b_x, norm)

def bc(p, q):
	'''
	Bhattacharyya distance
	'''
	return -np.log(np.sqrt(p * q).sum())


plt.plot(b_y.tolist()[:-1], [bc(signal_mass_dist, row) for row in pdfs][::-1], '-')
plt.xlabel(r'$\hat{y}$ (Deep Net output)')
plt.ylabel(r'Bhattacharyya distance $-\ln\left(\sum\sqrt{p_i q_i}\right)$')
plt.title(r"Similarity of QCD mass given $\hat{y}$ to $W' \rightarrow WZ$ mass.")
plt.savefig(PLOT_DIR % 'bhattacharyya-unweighted.pdf')
plt.show()



# -- weighted convergence

mass_weights = get_weights(np.random.uniform(0, 1100, 20000), mass[(signal == 0) & (mass < 1100)], 
	bins=np.linspace(0, 1100, 1000))




H, b_x, b_y = np.histogram2d(
	mass[(signal == 0) & (mass < 110)], 
	y_dl[(signal == 0) & (mass < 110)], 
	bins=(30, 10), 
	normed=True,
	weights=mass_weights[mass[(signal == 0) & (mass < 1100)] < 110])




plt.imshow(np.flipud(normalize_rows(H.T)), extent=(mass[(signal == 0)].min(), 110, 0, 1), aspect='auto', interpolation='none')
plt.xlabel('Jet Mass (GeV)')
plt.ylabel(r'$\hat{y}$ (Deep Net output)')
plt.title(r'PDF of QCD Jet Mass, binned in $\hat{y}$')
cb = plt.colorbar()
cb.set_label(r'$P(\mathrm{mass} \vert \hat{y})$')
plt.savefig(PLOT_DIR % 'mass-dist-yhat-weighted.pdf')
plt.show()



filt = data['image'][np.argsort(y_dl)[::-1][:3]].mean(axis=0)

filt = res['x'].reshape((25, 25))

abs_max = np.abs(filt.max())

plt.imshow(filt, cmap=cm.hot, interpolation='nearest')


def objective(im, model=dl):
	return 1 - dl.predict(im.reshape(1, 625)).ravel()[0]


def positive(x):
	return 1.0 * (np.abs(np.linalg.norm(x) - 1) > 1e-4)


ds = DescrStatsW(R[selection], weights=weights[selection])
S = ds.corrcoef
vm = np.max(np.abs(S)); plt.imshow(S, interpolation='nearest', cmap=cm, vmin=-vm, vmax=vm)

plt.title(r'Pairwise correlations of final layer node activations' + '\n' + 
	r"$W' \rightarrow WZ$ and QCD, Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [65, 95]$ GeV")
cb = plt.colorbar()
cb.set_label(r'Pearson Coefficient')
plt.xlabel('Node number')
plt.ylabel('Node number')
plt.savefig(PLOT_DIR % 'global-R-corrs.pdf')




ds = DescrStatsW(R[selection & (signal == 1)], weights=weights[selection & (signal == 1)])
S = ds.corrcoef
vm = np.max(np.abs(S)); plt.imshow(S, interpolation='nearest', cmap=cm, vmin=-vm, vmax=vm)

plt.title(r'Pairwise correlations of final layer node activations' + '\n' + 
	r"$W' \rightarrow WZ$ only, Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [65, 95]$ GeV")
cb = plt.colorbar()
cb.set_label(r'Pearson Coefficient')
plt.xlabel('Node number')
plt.ylabel('Node number')
plt.savefig(PLOT_DIR % 'wprime-R-corrs.pdf')
plt.show()

ds = DescrStatsW(R[selection & (signal == 0)], weights=weights[selection & (signal == 0)])
S = ds.corrcoef
vm = np.max(np.abs(S)); plt.imshow(S, interpolation='nearest', cmap=cm, vmin=-vm, vmax=vm)

plt.title(r'Pairwise correlations of final layer node activations' + '\n' + 
	r"QCD only, Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [65, 95]$ GeV")
cb = plt.colorbar()
cb.set_label(r'Pearson Coefficient')
plt.xlabel('Node number')
plt.ylabel('Node number')
plt.savefig(PLOT_DIR % 'qcd-R-corrs.pdf')
plt.show()







m_select = (mass > 80) & (mass < 95) & (signal == 1)


ds = DescrStatsW(R[selection & m_select], weights=weights[selection & m_select])
S = ds.corrcoef
vm = np.max(np.abs(S)); plt.imshow(S, interpolation='nearest', cmap=cm, vmin=-vm, vmax=vm)

plt.title(r'Pairwise correlations of final layer node activations' + '\n' + 
	r"$W' \rightarrow WZ$ only, Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [80, 95]$ GeV")
cb = plt.colorbar()
cb.set_label(r'Pearson Coefficient')
plt.xlabel('Node number')
plt.ylabel('Node number')
plt.savefig(PLOT_DIR % 'wprime-R-corrs-80-95.pdf')
plt.show()



mass_bins = [65, 70, 75, 80, 85, 90, 95]

plt.figure(figsize=(18, 15)) 
for label, m_min, m_max in enumerate(zip(mass_bins, mass_bins[1:])):
# for label in np.unique(y_test):
    plt.subplot(3, 4, label + 1)
    plt.title("$W' \rightarrow WZ$ encodings, Jet $p_T\in[250, 300]$ $\mathsf{GeV},\vert\eta\vert<2$, $m_{\mathsf{jet}}\in [%i, %i]$ GeV" % (m_min, m_max))
    encodings = R[selection & (mass < m_max) & (mass > m_min)]
    
    # encodings is nexamples x 10
    means = np.mean(encodings, axis=0)
    stds = np.std(encodings, axis=0)
    
    bar_centers = np.arange(X_test_repr.shape[1])
    pl.bar(bar_centers, means, width=0.8, align='center', yerr=stds, alpha=0.5)
    pl.xticks(bar_centers, bar_centers)
    pl.xlim((-0.5, bar_centers[-1] + 0.5))
    #pl.ylim((0, 0.3))




