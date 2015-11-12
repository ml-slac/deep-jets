'''
performance.py
author: Luke de Oliveira (lukedeo@stanford.edu)


Usage:

>>> weights = np.ones(n_samples)
>>> # -- going to match bkg to signal
>>> weights[signal == True] = get_weights(sig_pt, bkg_pt)
>>> discs = {}
>>> add_curve(r'\tau_{32}', 'red', calculate_roc(signal, tau_32, weights=weights))
>>> fg = ROC_plotter(discs)
>>> fg.savefig('myroc.pdf')

'''

import numpy as np
import matplotlib.pyplot as plt

def get_weights(target, actual, bins = 10, cap = 10, match = True):
	'''
	re-weights a actual distribution to a target.

	Args:
		target (array/list): observations drawn from target distribution
		actual (array/list): observations drawn from distribution to 
			match to the target.

		bins (numeric or list/array of numerics): bins to use to do weighting

		cap (numeric): maximum weight value.

		match (bool): whether to make the sum of weights in actual equal to the
			number of samples in target

	Returns:
		numpy.array: returns array of shape len(actual).

	'''
	target_counts, target_bins = np.histogram(target, bins=bins)
	counts, _ = np.histogram(actual, bins=target_bins)
	counts = (1.0 * counts)
	counts = np.array([max(a, 0.0001) for a in counts])
	multiplier = np.array((target_counts / counts).tolist() + [1.0])

	weights = np.array([min(multiplier[target_bins.searchsorted(point) - 1], cap) for point in actual])
	# weights = np.array([target_bins.searchsorted(point) for point in actual])

	if match:
		weights *= (len(target) / np.sum(weights))

	return weights


def calculate_roc(labels, discriminant, weights=None, bins = 2000):
	'''
	makes a weighted ROC curve

	Args:
		labels (numpy.array): an array of 1/0 representing signal/background
		discriminant (numpy.array): an array that represents the discriminant
		weights: sample weights for each point. 
			`assert(weights.shape == discriminant.shape)
		bins: binning to use -- can be an int or a list/array of bins.

	Returns:
		tuple: (signal_efficiency, background_rejection) where each are arrays

	'''
	sig_ind = labels == 1
	bkg_ind = labels == 0
	if weights is None:
		bkg_total = np.sum(labels == 0)
		sig_total = np.sum(labels == 1)
	else:
		bkg_total = np.sum(weights[bkg_ind])
		sig_total = np.sum(weights[sig_ind])

	discriminant_bins = np.linspace(np.min(discriminant), np.max(discriminant), bins)

	if weights is None:
		sig, _ = np.histogram(discriminant[sig_ind], discriminant_bins)
		bkd, _ = np.histogram(discriminant[bkg_ind], discriminant_bins)
	else:
		sig, _ = np.histogram(discriminant[sig_ind], discriminant_bins, weights = weights[sig_ind])
		bkd, _ = np.histogram(discriminant[bkg_ind], discriminant_bins, weights = weights[bkg_ind])

	sig_eff = np.add.accumulate(sig[::-1]) / float(sig_total)
	bkg_rej = 1 / (np.add.accumulate(bkd[::-1]) / float(bkg_total))

	return sig_eff, bkg_rej





def ROC_plotter(curves, min_eff = 0, max_eff = 1, linewidth = 1.4, 
	pp = False, signal = "$Z\rightarrow t\bar{t}$", background = "QCD", 
	title = "Jet Image Tagging Comparison", logscale = True, ymax=10**4, ymin=1):	

	fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
	ax = fig.add_subplot(111)
	plt.xlim(min_eff,max_eff)
	plt.grid(b = True, which = 'minor')
	plt.grid(b = True, which = 'major')
	max_ = 0
	for tagger, data in curves.iteritems():
		sel = (data['efficiency'] >= min_eff) & (data['efficiency'] <= max_eff)
		if np.max(data['rejection'][sel]) > max_:
			max_ = np.max(data['rejection'][sel])
		plt.plot(data['efficiency'][sel], data['rejection'][sel], '-', label = r''+tagger, color = data['color'], linewidth=linewidth)

	ax = plt.subplot(1,1,1)
	for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
		item.set_fontsize(20) 
	if logscale == True:	
		plt.ylim(ymin,ymax)
		ax.set_yscale('log')
	ax.set_xlabel(r'$\epsilon_{\mathrm{signal}}$')
	ax.set_ylabel(r"$1 / \epsilon_{\mathrm{bkg}}$")

	plt.legend()
	plt.title(r''+title)
	if pp:
		pp.savefig(fig)
	else:
		plt.show()
		return fig


def add_curve(name, color, curve_pair, dictref):
	dictref.update(
		{
			name : {
						'efficiency' : curve_pair[0], 
						'rejection' : curve_pair[1], 
						'color' : color
					}
		}
	)







