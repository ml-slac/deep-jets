# -- TRAINING for full [200, 400] window
from keras.layers import containers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder, MaxoutDense, Activation, Merge
from keras.layers.advanced_activations import PReLU
from keras.layers.embeddings import Embedding
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

%run ./viz/visualize.py
%run ./viz/performance.py
%run ./dl/deepae.py
%run ./utils/sampling.py
%run ./likelihood.py

PLOT_DIR = './plots/8-27/%s'


# dat = np.load('../deep-jets/data/processed/wprime-800-qcd-8-17.npy')
dat = np.load('../jet-simulations/slac-data.npy')
print '{} jets before preselection'.format(dat.shape[0])


# -- apply preselection
# dat = dat[(dat['jet_pt'] > 20) & (dat['jet_pt'] < 300)]
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

signal, pt, mass, tau_21 = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

X_ = np.array([x.ravel() for x in data['image']]).astype('float32')
# X_ = X_ * data['total_energy'][:, np.newaxis]
y_ = data['signal'].astype('float32')



n_train = int(0.7 * data.shape[0])

print 'Using {} training points, and {} testing points.'.format(n_train, data.shape[0] - n_train)

X_train, y_train, X_test, y_test = X_[:n_train], y_[:n_train], X_[n_train:], y_[n_train:]




# -- kinematic selection


# signal, pt, mass, tau_21,  = data['signal'], data['jet_pt'], data['jet_mass'], data['tau_21']

# -- plot some kinematics...
plt.hist(pt[signal == True], bins=np.linspace(200, 400, 100), 
	histtype='step', color='red', label=r"$W' \rightarrow WZ$", linewidth=2)
plt.hist(pt[signal == False], bins=np.linspace(200, 400, 100), 
	histtype='step', color='blue', label='QCD', linewidth=2)
plt.xlabel(r'$p_T$ [GeV]')
plt.ylabel('Count')
plt.ylim(0, 60000)
plt.title(r'Jet $p_T$ distribution, $p_T^W \in [200, 400]$ GeV' + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR % 'unweighted-pt-distribution.pdf')
plt.show()


# -- calculate the weights
weights = np.ones(data.shape[0])

reference_distribution = np.random.uniform(200, 400, signal.sum())

weights[signal == 1] = get_weights(reference_distribution, pt[signal == 1], 
	bins=np.linspace(200, 400, 200))

weights[signal == 0] = get_weights(reference_distribution, pt[signal != 1], 
	bins=np.linspace(200, 400, 200))
# weights[signal == 1] = get_weights(pt[signal != 1], pt[signal == 1], 
# 	bins=np.concatenate((
# 		np.linspace(200, 300, 1000), np.linspace(300, 1005, 500)))
# 	)



# -- plot reweighted...
plt.hist(pt[signal == True], bins=np.linspace(200, 400, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1], linewidth=2)
plt.hist(pt[signal == False], bins=np.linspace(200, 400, 100), histtype='step', color='blue', label='QCD', weights=weights[signal == 0], linewidth=2)
plt.xlabel(r'$p_T$ (GeV)')
plt.ylabel('Count')
plt.title(r'Weighted Jet $p_T$ distribution (matched to QCD)')
plt.legend()
         
plt.savefig(PLOT_DIR % 'weighted-pt-distribution.pdf')
plt.show()



# -- plot weighted mass
plt.hist(mass[signal == True], bins=np.linspace(65, 95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1], linewidth=2)
plt.hist(mass[signal == False], bins=np.linspace(65, 95, 100), histtype='step', color='blue', label='QCD', weights=weights[signal == 0], linewidth=2)
plt.ylim(0, 82000)
plt.xlabel(r'Jet $m$ [GeV]')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m$ distribution ($p_T^W \in [200, 400]$ GeV flattened)' + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-mass-distribution.pdf')
plt.show()



# -- plot weighted tau_21
plt.hist(tau_21[signal == True], bins=np.linspace(0, 0.95, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1], linewidth=2)
plt.hist(tau_21[signal == False], bins=np.linspace(0, 0.95, 100), histtype='step', color='blue', label='QCD', weights=weights[signal == 0], linewidth=2)
plt.ylim(0, 80000)
plt.xlabel(r'Jet $\tau_{21}$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $\tau_{21}$ distribution ($p_T^W \in [200, 400]$ GeV flattened)' + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-tau21-distribution.pdf')
plt.show()

# P_mass = Likelihood1D(np.concatenate((np.array([0, 25]), np.linspace(25, 35, 5), np.linspace(35, 160, 20), np.array([160, 900]))))
# P_mass.fit(mass[:n_train][signal[:n_train] == 1], mass[:n_train][signal[:n_train] == 0], weights=(weights[:n_train][signal[:n_train] == 1], weights[:n_train][signal[:n_train] == 0]))
# mass_likelihood = P_mass.predict(mass)

# # -- plot weighted mass likelihood
# plt.hist((mass_likelihood[signal == True]), bins=np.linspace(0, 10, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
# plt.hist((mass_likelihood[signal == False]), bins=np.linspace(0, 10, 100), histtype='step', color='blue', label='QCD')
# plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
# plt.ylabel('Count')
# plt.title(r'Weighted Jet $m$ likelihood distribution ($p_T$ matched to QCD)')
# plt.legend()
# plt.savefig(PLOT_DIR % 'weighted-mass-likelihood-distribution.pdf')
# plt.show()




mass_bins = np.linspace(64.99, 95.01, 10)#, np.linspace(35, 159.99, 30), np.array([160, 900])))
tau_bins = np.concatenate((np.array([0, 0.1]), np.linspace(0.1001, 0.7999999999, 10), np.array([0.8, 1])))
P_2d = Likelihood2D(mass_bins, tau_bins)
P_2d.fit((mass[signal == 1], tau_21[signal == 1]), (mass[signal == 0], tau_21[signal == 0]), weights=(weights[signal == 1], weights[signal == 0]))
P_2d.fit((mass[signal == 1], tau_21[signal == 1]), (mass[signal == 0], tau_21[signal == 0]), weights=(weights[signal == 1], weights[signal == 0]))
mass_nsj_likelihood = P_2d.predict((mass, tau_21))
log_likelihood = np.log(mass_nsj_likelihood)
# -- plot weighted mass + nsj likelihood
plt.hist((log_likelihood[signal == True]), bins=np.linspace(-3.6, 6.5, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1])
plt.hist((log_likelihood[signal == False]), bins=np.linspace(-3.6, 6.5, 100), histtype='step', color='blue', label='QCD')
plt.xlabel(r'$P(\mathrm{signal}) / P(\mathrm{background})$')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m, \tau_{21}$ likelihood distribution ($p_T$ matched to QCD)')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-mass-nsj-likelihood-distribution.pdf')
plt.show()



y_dl = np.load('./yhat.npy')

# -- plot DL output
plt.hist(y_dl[signal == True], bins=np.linspace(0, 1, 100), histtype='step', color='red', label=r"$W' \rightarrow WZ$", weights=weights[signal == 1], linewidth=2)
plt.hist(y_dl[signal == False], bins=np.linspace(0, 1, 100), histtype='step', color='blue', label='QCD', weights=weights[signal == 0], linewidth=2)
# plt.ylim(0, 82000)
plt.xlabel(r'Deep Network Output')
plt.ylabel('Count')
plt.title(r'Weighted Jet $m$ distribution ($p_T^W \in [200, 400]$ GeV flattened)' + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
plt.legend()
plt.savefig(PLOT_DIR % 'weighted-deep-net-distribution.pdf')
plt.show()




# ROC curves

discs = {}
add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal, 2-tau_21, weights=weights), discs)
add_curve(r'Deep Network, trained on $p_T^W \in [200, 400]$ GeV', 'red', calculate_roc(signal, y_dl, weights=weights), discs)
add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood)', 'blue', calculate_roc(signal, (log_likelihood), bins=1000000, weights=weights), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD ($p_T^W \in [200, 400]$ GeV flattened)" + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
fg.savefig(PLOT_DIR % 'roc.pdf')

plt.show()






pt_selection = (pt > 250) & (pt < 300)
selection = pt_selection

# ROC curves

discs = {}
add_curve(r'$\tau_{21}$', 'black', calculate_roc(signal[selection], 2-tau_21[selection], weights=weights[selection]), discs)
add_curve(r'Deep Network, trained on $p_T^W \in [200, 400]$ GeV', 'red', calculate_roc(signal[selection], y_dl[selection], weights=weights[selection]), discs)
add_curve(r'$m_{\mathrm{jet}}, \tau_{21}$ (2D likelihood)', 'blue', calculate_roc(signal[selection], (log_likelihood[selection]), bins=1000000, weights=weights[selection]), discs)
fg = ROC_plotter(discs, title=r"$W' \rightarrow WZ$ vs. QCD (trained on $p_T^W \in [200, 400]$ GeV flattened)" + '\n' + 
	r'$m_{W}\in [65, 95]$ GeV')
fg.savefig(PLOT_DIR % 'roc.pdf')

plt.show()



# -- build the model
dl = Sequential()
# dl.add(Merge([raw, gaussian], mode='concat'))
# dl.add(Dense(1000, 512))
dl.add(Dropout(0.1))
dl.add(Dense(625, 512, W_regularizer=regularizers.l2(0.005)))

dl.add(Dropout(0.1))
dl.add(Dense(512, 256, W_regularizer=regularizers.l2(0.005)))
# dl.add(Activation('tanh'))

dl.add(Dropout(0.1))
dl.add(Dense(256, 64))

dl.add(Dropout(0.1))
dl.add(Dense(64, 25))

dl.add(Dropout(0.1))
dl.add(Dense(25, 1))
dl.add(Activation('sigmoid'))

dl.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')

h = dl.fit(X_train, y_train, batch_size=512, nb_epoch=20, show_accuracy=True, 
               validation_split=0.25, 
               callbacks = [
                   EarlyStopping(verbose=True, patience=6, monitor='val_loss'),
                   ModelCheckpoint('./SLACNet[200-400]-flat-weights.h5', monitor='val_loss', verbose=True, save_best_only=True)
               ], 
               sample_weight=weights[:n_train])

y_dl = dl.predict(X_, verbose=True).ravel()









