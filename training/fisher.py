"""
The module implements Fisher Discriminant Analysis.
"""
__author__ = 'Michael Kagan mkagan@cern.ch'
#
# Code based on sklearn LDA code written by: Matthieu Perrot
#                                            Mathieu Blondel
#
# using algorithms as described in:
# Zhang, et. al. 'Regularized Discriminant Analysis, Ridge Regression and Beyond' Journal of Machine Learning Research 11 (2010) 2199-2228
#
 
import warnings
import sys
import time

import numpy as np
from scipy import linalg

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import logsumexp
# from sklearn.utils.fixes import unique
from numpy import unique
from sklearn.utils import check_array
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels

__all__ = ['Fisher', 'KernelFisher']


#####################################################################################################################
#NOTE TO SELF:
# np.inner(A,B) sums over last indices, i.e. = A[i,j]*B[k,j]
# so if you want to do A*B, you should do np.inner(A, B.T)
# Also, np.inner is faster than np.dot
#####################################################################################################################


class Fisher(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Fisher Discriminant Analysis (LDA)

    A classifier with a linear decision boundary, generated
    by fitting class conditional densities to the data
    fisher criteria of maximizing between class variance
    while minimizing within class variance

    The fitted model can also be used to reduce the dimensionality
    of the input, by projecting it to the most discriminative
    directions.

    Parameters
    ----------

    norm_covariance :  boolean
        if true, the covariance of each class will be divided by (n_points_in_class - 1)

    n_components: int
        Number of components (< n_classes - 1) for dimensionality reduction

    priors : array, optional, shape = [n_classes]
        Priors on classes

    Attributes
    ----------
    `means_` : array-like, shape = [n_components_found_, [n_classes, n_features] ]
        Class means, for each component found
    `w_` : array-like, shape = [n_components_found_, n_features ]
        decision vector, for each component found
    `priors_` : array-like, shape = [n_classes]
        Class priors (sum to 1)
    `covs_` : array, shape = [n_components_found_, [ [n_features, n_features], [n_features, n_features] ] one cov for class=0 and one for class=1
        Covariance matrix (shared by all classes)
    `n_components_found_` : int
        number of fisher components found, which is <= n_components
        
    Examples (put fisher.py in working directory)
    --------
    >>> import numpy as np
    >>> from fisher import Fisher
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> fd = Fisher()
    >>> fd.fit(X, y)
    Fisher(n_components=1, norm_covariance=True, priors=None)
    >>> print(fd.transform([[-0.8, -1]]))
    [[-1.]]


    """

    def __init__(self, norm_covariance = True, n_components=None, priors=None):
        self.norm_covariance = norm_covariance
        self.n_components = 1 if n_components==None else n_components
        self.priors = np.asarray(priors) if priors is not None else None

        if self.priors is not None:
            if (self.priors < 0).any():
                raise ValueError('priors must be non-negative')
            if self.priors.sum() != 1:
                print 'warning: the priors do not sum to 1. Renormalizing'
                self.priors = self.priors / self.priors.sum()


    def fit(self, X, y, store_covariance=False, tol=1.0e-4,
            do_smooth_reg=False, cov_class=None, cov_power=1):
        """
        Fit the Fisher Discriminant model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers)
        store_covariance : boolean
            If True the covariance matrix of each class and each iteration is computed
            and stored in `self.covs_` attribute. has dimensions [n_iterations][2] where 2 is for nclasses = 2
        tol:  float
            used for regularization, either for svd series truncation or smoothing.
        do_smooth_reg: boolean
            If False, truncate SVD matrix inversion for singular values less then tol.
            If True, apply smooth regularization (filter factor) on inversion, such that 1/s_i --> s_i/(s_i^2 + tol^2), where s_i is singular value
        """
        # X, y = check_array(X, y, sparse_format='dense')
        X, y = check_array(X), check_array(y)#, sparse_format='dense')
        self.classes_, y = unique( (y>0), return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        self.means_ = []
        self.covs_  = []
        
        wvecs = []

        # Group means n_classes*n_features matrix

        means = []
        nevt = np.zeros(n_classes)
        Xc = []
        Xg = []
        covs = []
        cov = None
            
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            nevt[ind] = Xg.shape[0]
                
        # centered group data
            if cov_class is None or cov_class == ind:
                Xgc = Xg - meang
                covg = np.zeros((n_features, n_features))
                covg += np.dot(Xgc.T, Xgc)
                covs.append(covg)
             

        # check rank of Sb = m * m.T
        # if rank = 0, we are in null space of Sb, and can not calculate fisher component
        m = means[0] - means[1]
        if linalg.norm(m) ==0:
            print "WARNING: Inter-class matrix is zero, i.e. classes have same mean!"
            print "         Fisher can not discriminate in this case --> Exiting"
            sys.exit(2)
            
        Sb = np.outer( m, m )
        #svdvalsSb = linalg.svdvals( Sb )
        #rank = np.sum( svdvalsSb > tol )
        #print "rank Sb = ",rank            

        self.means_.append( np.asarray(means) )

        #covs_array = [ np.asarray(covs[0]) , np.asarray(covs[1]) ]
        covs_array = [np.asarray(cc) for cc in covs]
        if self.norm_covariance:
            for ii in range(len(covs_array)):
                covs_array[ii] /= ( (nevt[ii]-1.0) if nevt[ii] > 1 else 1 )
#            covs_array[0] /= ( (nevt[0]-1.0) if nevt[0] > 1 else 1 )
#            covs_array[1] /= ( (nevt[1]-1.0) if nevt[1] > 1 else 1 )

        if store_covariance:
            self.covs_.append( covs_array )

        #if norm_covariance:
        #    nevt[0] = nevt[0] if nevt[0] > 1 else 2
        #    nevt[1] = nevt[1] if nevt[1] > 1 else 2
        #    self.covs_.append( [ np.asarray(covs[0]) / (nevt[0]-1.0), np.asarray(covs[1]) / (nevt[1]-1.0) ] )
        #else:
        #    self.covs_.append( [ np.asarray(covs[0]), np.asarray(covs[1]) ] )

        #Sw = covs_array[0] + covs_array[1]
        Sw = sum(covs_array)

        #----------------------------
        # for 2 class system, need to solve for w in
        # Sb * w = lambda * Sw * w
        # where lambda is eigenvalue of this generalized eigenvalue problem
        # however, Sb * w = m mT * w = m * constant
        # implies we only need to solve m = Sw * w   
        # (overall constant wet later with ||w||=1 )
        # solution: Sw = U*S*Vh using svd ==> S.inv*U.T*m = Vh *w ==> w = Sum_i^rank(S) vh_i * (U.T * m)_i / S_i
        # where vh_i is a vector
        #----------------------------
        # step 1)  svd of Sw
        # step 2) calculate sum for all non singular components
        U, S, V = linalg.svd(Sw)        

        rank = np.sum(S > tol)
        #print "rank Sw = ", rank

        S = np.power(S, cov_power)
       
        UTm = np.inner(U.T, m)
        w = np.zeros(n_features)
        for i in range(len(S)):
            if do_smooth_reg==True:
                w += V[i,:] * UTm[i] * ( S[i] / (S[i]*S[i]+ tol**(2*cov_power)) )
                #w += V[i,:] * UTm[i] * ( S[i] / (S[i]*S[i] + tol*tol) )
            else:
                if S[i] < tol: 
                    continue
                w += V[i,:] * UTm[i] / S[i]

        if linalg.norm(w) != 0:
            w /= linalg.norm(w)
        else:
            print "WARNING: Fisher discriminant line has norm=0 --> no discriminating curved found! Exiting"
            sys.exit(2)
            
        #check if signal (1) projection smaller than bkg (0), if so, add minus sign
        if(np.inner(means[1],w) < np.inner(means[0],w)):
            w *= (-1.0)

        wvecs.append( w ) 

        
        self.w_ = np.asarray(wvecs)
        self.n_components_found_ = len(self.w_)
        self.singular_vals = S

        return self


    def fit_multiclass(self, X, y, use_total_scatter=False, solution_norm="N", sigma_sqrd=1e-8, tol=1.0e-3, print_timing=False):
        """
        Fit the Fisher Discriminant model according to the given training data and parameters.
        Based on (but depending on options not exactly the same as) "Algorithm 4" in
        Zhang, et. al. 'Regularized Discriminant Analysis, Ridge Regression and Beyond' Journal of Machine Learning Research 11 (2010) 2199-2228
        NOTE: setting norm_covariance=False and use_total_scatter=True, and solution_norm = 'A' or 'B' will give the algorithm from paper

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array, shape = [n_samples]
            Target values (integers)
        use_total_scatter : boolean
            If True then use total scatter matrix St = Sum_i (x_i - m)(x_i - m).T instead of Sw
            If False, use Sw = Sum_{c=1... n_classes} Sum_{i; x in class c} norm_c (x_i - m_c)(x_i - m_c).T
                      where norm_c = 1/N_samples_class_c if norm_covariance=True, else norm_c = 1
        solution_norm: boolean
            3 kinds of norms, "A", "B", or "N", were "N" means normalize to 1.  "A" and "B" (see paper reference) have normalizations
            that may be important when consitering n_classes > 2
        sigma_sqrd:  float
            smooth regularization parameter, which is size of singular value where smoothing becomes important.
            NOTE: is fraction in case norm_covariance=False, as a priori the scale of the singular values is not known in this case
        tol:  float
            used for truncated SVD of Sw.  Essentially a form of regularization.  Tol for SVD(R) is 1e-6, fixed right now
        print_timing: boolean
            print time for several matrix operations in the algorithm
        """
        # X, y = check_array(X, y, sparse_format='dense')
        X, y = check_array(X), check_array(y)#, sparse_format='dense')
        self.classes_, y = unique( y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        n_samples_perclass = np.bincount(y)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        if not any( np.array(["A","B","N"])==solution_norm ):
             print 'WARNING: solution_norm must be one of ["A","B","N"]! Exiting'
             sys.exit(2)

        ts = time.time()
                    
        self.means_ = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            self.means_.append(np.asarray(meang))
        if print_timing: print 'fit_multiclass: means took', time.time() - ts

        ts = time.time()
        PI_diag = np.diag( 1.0*n_samples_perclass )                                       # shape(PI_diag) = n_classes x n_classes
        PI_inv = np.diag( 1.0 / (1.0*n_samples_perclass) )                                # shape(PI_inv) = n_classes x n_classes
        PI_sqrt_inv = np.sqrt( PI_inv )                                                   # shape(PI_sqrt_inv) = n_classes x n_classes
        #H = np.identity(n_samples) - (1.0/(1.0*n_samples))*np.ones((n_samples,n_samples))
        E=np.zeros( (n_samples,n_classes) )
        E[[range(n_samples),y]]=1
        if print_timing: print 'fit_multiclass: matrices took', time.time() - ts


        ts = time.time()
        #note: computation of this is fast, can always do it inline, if memory consumption gets large
        Xt_H = X.T - (1.0/(1.0*n_samples))*np.repeat( np.array([X.T.sum(1)]).T, n_samples, axis=1)    # shape(Xt_H) = n_features x n_samples
        if print_timing: print 'fit_multiclass: Xt_H took', time.time() - ts

        ts = time.time()
        #####################################################################################################################
        #Sb = X.T * H * E * PI_inv * E.T * H * X = (X.T * H * E * PI_sqrt_inv) * (X.T * H * E * PI_sqrt_inv).T
        #if norm_covariance: Sb = X.T * H * E * PI_inv * PI_inv * E.T * H * X = (X.T * H * E * PI_inv) * (X.T * H * E * PI_inv).T
        #This norm actually doesn't matter in 2-class, I think it jsut becomes an overall scaling, which gets normalized away
        #I expect id doesn't matter for multiclass either... but not sure
        #to be clear, multi-class fisher does not norm! but then its harder to set the regularization factor for Sw
        #####################################################################################################################

        Xt_H_E_PIsi = None                                                      # shape(Xt_H_E_PIsi) = n_features x n_classes
        if self.norm_covariance:
           Xt_H_E_PIsi =  np.dot(Xt_H, np.dot(E, PI_inv) )
        else:
           Xt_H_E_PIsi = np.dot(Xt_H, np.dot(E, PI_sqrt_inv) )
        if print_timing: print 'fit_multiclass: Xt_H_E_PIsi took', time.time() - ts

        
        #St_reg = ( np.dot(X.T np.dot(H, X)) - (sigma*sigma)*np.identity(n_features))

        ts = time.time()
        #####################################################################################################################
        #Sw = X.T * [ 1 - E*PI_inv*E.T ] * X = X.T * X - M.T * PI * M
        # if norm_covariance: Sw = X.T * [ P - E*PI_inv*PI_inv*E.T ] * X = X.T *P * X - M.T * M
        #####################################################################################################################
        M = np.asarray(self.means_)                                              # shape(M) = n_classes x n_features
        #P = np.diag( np.dot(E, 1.0/(1.0*n_samples_perclass)) )
        P_vec = np.array([np.dot(E, 1.0/(1.0*n_samples_perclass))]).T            # shape(P_vec) = n_samples x 1
        Sw=None                                                                  # shape(Sw) = n_features x n_features 
        if not use_total_scatter:
            if self.norm_covariance:
                #Sw = np.inner( np.inner(X.T, P), X.T) - np.dot( M.T, M)
                Sw = np.inner( (P_vec*X).T, X.T) - np.dot( M.T, M)
            else:
                Sw = np.inner(X.T, X.T) - np.dot( M.T, np.dot(PI_diag, M))
                
            if print_timing: print 'fit_multiclass: Sw took', time.time() - ts

        #####################################################################################################################
        #assume (I think true) for condensed svd, where we only take vectors for non-zero singular values
        #that if M is symmetric, then Uc=Vc where condensed_svd(M) = Uc * Sc * Vc.T
        #this is because the singular values of a symmetric matrix are the abosolute values of the non-zero eigenvalues
        #so assuming the singular vectors of the non-zero singular values are the same as eigen vectors
        #and since condensed svd only keeps singular vectors for non-zero singular values, should have Uc==Vc
        #####################################################################################################################


        ts = time.time()
        Uc, Sc, Utc, Sc_norm = None, None, None, None
        if use_total_scatter:
            St_norm = (1.0/(1.0*n_samples)) if self.norm_covariance else 1.0
            Uc, Sc, Utc, Sc_norm = self.condensed_svd( St_norm * np.inner(Xt_H, X.T), tol, store_singular_vals=True )
        else:
            Uc, Sc, Utc, Sc_norm = self.condensed_svd( Sw, tol, store_singular_vals=True )
        if print_timing: print 'fit_multiclass: Uc, Sc, Utc took', time.time() - ts

        ts = time.time()
        #scale up sigma to appropriate range of singular values
        reg_factor = sigma_sqrd * Sc_norm 
        St_reg_inv = np.dot( Uc, np.dot(np.diag(1.0/(Sc + reg_factor)), Utc) )    # shape(St_reg_inv) = n_features x n_features
        if print_timing: print 'fit_multiclass: St_reg_inv took', time.time() - ts

        ts = time.time()
        G = np.dot(St_reg_inv, Xt_H_E_PIsi)                                       # shape(G) = n_features x n_classes
        if print_timing: print 'fit_multiclass: G took', time.time() - ts

        ts = time.time()
        R = np.dot( Xt_H_E_PIsi.T, G)                                             # shape(R) = n_classes x n_classes
        if print_timing: print 'fit_multiclass: R took', time.time() - ts

        ts = time.time()
        Vr, Lr, Vtr, Lr_norm =  self.condensed_svd( R, tol=1e-6 )                 # shape(Vr) = n_classes x rank_R
        if print_timing: print 'fit_multiclass: Vr, Lr, Vtr took', time.time() - ts
        
        ts = time.time()
        W = np.dot( G, Vr)                                                        # shape(W) = n_features x rank_R
        if print_timing: print 'fit_multiclass: B took', time.time() - ts
        
        if solution_norm=="A":
            W = np.dot(W, np.diag(1.0 / np.sqrt(Lr)) )

        elif solution_norm=="N":
            for i in range( W.shape[1] ):
                if linalg.norm(W[:,i]) != 0:
                    W[:,i] /= linalg.norm(W[:,i])
                else:
                    print "WARNING: Fisher discriminant line has norm=0 --> no discriminating curved found! Exiting"
                    sys.exit(2)

        
        self.w_ = W.T  #transpose here just because want to store the matrix where rows have length n_features, i.e. are discriminants 

        return self

    def condensed_svd(self, M, tol=1e-3, store_singular_vals=False):
        U, S, Vt = linalg.svd(M, full_matrices=False)

        if store_singular_vals:
            self.singular_vals = S

        #want tolerance on fraction of variance in singular value
        #when not norm_covariance, need to normalize singular values
        S_norm = 1.0 if self.norm_covariance else np.sum(S)

        rank = np.sum( (S/S_norm) > tol )

        return U[:,:rank], S[:rank], Vt[:rank,:], S_norm


    @property
    def classes(self):
        warnings.warn("Fisher.classes is deprecated and will be removed in 0.14. "
                      "Use .classes_ instead.", DeprecationWarning,
                      stacklevel=2)
        return self.classes_

    def _decision_function(self, X):
        X = np.asarray(X)
        # center and scale data
        #X = np.dot(X - self.xbar_, self.scaling)
        #return np.dot(X, self.coef_.T) + self.intercept_
        return np.inner( X, self.w_ )

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples, n_components_found_]
            Decision function values related to each class, per sample
            n_components_found_ is the number of components requested and found
            even if n_components_found_=1, a 2D array is found, 
            but can be promoted to 1D array with dimension [n_samples] with decision_function(X)[:,0]
        """
        dec_func = self._decision_function(X)
        #if len(self.w_) == 1:
        #    return dec_func[:, 0]
        return dec_func

    def transform(self, X):
        """
        Project the data so as to maximize class separation (large separation
        between projected class means and small variance within each class).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components_found_]
        """
        X = np.asarray(X)
        # center and scale data
        #X = np.dot(X - self.xbar_, self.scaling)
        #n_comp = X.shape[1] if self.n_components is None else self.n_components
        #return np.dot(X, self.coef_[:n_comp].T)
        dec_func = self._decision_function(X)
        return dec_func

    def fit_transform(self, X, y, store_covariance=False, tol=1.0e-4):
        """
        Fit the Fisher Discriminant model according to the given training data and parameters.
        The project the data onto up to n_components so as to maximize class separation (large separation
        between projected class means and small variance within each class).
        NOTE this function is not clever, it simply runs fit(X,y [, store_covariance, tol]).transform(X)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array, shape = [n_samples]
            Target values (integers)
        store_covariance : boolean
            If True the covariance matrix of each class and each iteration is computed
            and stored in `self.covs_` attribute. has dimensions [n_iterations][2] where 2 is for nclasses = 2

        Returns
        -------
        X_new : array, shape = [n_samples, n_components_found_]
        """
        return self.fit(X, y, store_covariance, tol).transform(X)



########################################################################
########################################################################
########################################################################
########################################################################



class KernelFisher(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Kernalized Fisher Discriminant Analysis (KDA)

    A classifier with a non-linear decision boundary, generated
    by fitting class conditional densities to the data
    fisher criteria of maximizing between class variance
    while minimizing within class variance.

    The fisher criteria is used in a non-linear space, by transforming
    the data, X, of dimension D onto a D-dimensional manifold of
    a D' dimensional space (where D' is possible infinite) using a funtion f(X).
    The key to solving the problem in the non-linear space is to write
    the solution to fisher only in terms of inner products of
    the vectors X*Y.  Then the kernel trick can be employed, such that
    the standard inner product is promoted to a general inner product.
    That is, K(X,Y) = X*Y --> K(X,Y) = f(X)*f(Y), which is allowed for
    valid Kernels.  In this case, the function f() does not need to be
    known, but only the kernel K(X,Y).

    The fitted model can also be used to reduce the dimensionality
    of the input, by projecting it to the most discriminative
    directions.

    Parameters
    ----------

    use_total_scatter : boolean
        If True then use total scatter matrix St = Sum_i (x_i - m)(x_i - m).T instead of Sw
        If False, use Sw = Sum_{c=1... n_classes} Sum_{i; x in class c} norm_c (x_i - m_c)(x_i - m_c).T
                   where norm_c = 1/N_samples_class_c if norm_covariance=True, else norm_c = 1

    sigma_sqrd:  float
        smooth regularization parameter, which is size of singular value where smoothing becomes important.
        NOTE: is fraction in case norm_covariance=False, as a priori the scale of the singular values is not known in this case

    tol:  float
         used for truncated SVD of St.  Essentially a form of regularization.  Tol for SVD(R) is 1e-6, fixed right now

    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel used for generalized inner product.
        Default: "linear"

    degree : int, optional
        Degree for poly
        Default: 3.

    gamma : float, optional
        Kernel coefficient for rbf, sigmoid and poly kernels.
        Default: 1/n_features.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.

    norm_covariance :  boolean
        if true, the covariance of each class will be divided by (n_points_in_class - 1)
        NOTE: not currently used

    priors : array, optional, shape = [n_classes]
        Priors on classes

    print_timing: boolean
        print time for several matrix operations in the algorithm

    Attributes
    ----------
    `means_` : array-like, shape = [n_components_found_, [n_classes, n_features] ]
        Class means, for each component found
    `priors_` : array-like, shape = [n_classes]
        Class priors (sum to 1)
    
    `n_components_found_` : int
        number of fisher components found, which is <= n_components
        
    Examples (put fisher.py in working directory)
    --------
    >>> import numpy as np
    >>> from fisher import KernelFisher
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> fd = KernelFisher()
    >>> fd.fit(X, y)
    KernelFisher(coef0=1, degree=3, gamma=None, kernel='linear',
       norm_covariance=False, print_timing=False, priors=None,
       sigma_sqrd=1e-08, tol=0.001, use_total_scatter=True)
    >>> print(fd.transform([[-0.8, -1]]))
    [[-7.62102356]]]

    """

    def __init__(self, use_total_scatter=True, sigma_sqrd=1e-8, tol=1.0e-3,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 norm_covariance = False, priors=None, print_timing=False):

        self.use_total_scatter = use_total_scatter
        self.sigma_sqrd = sigma_sqrd
        self.tol = tol
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self._centerer = KernelCenterer()

        self.norm_covariance = norm_covariance
        self.print_timing = print_timing
        
        
        self.priors = np.asarray(priors) if priors is not None else None
        
        if self.priors is not None:
            if (self.priors < 0).any():
                raise ValueError('priors must be non-negative')
            if self.priors.sum() != 1:
                print 'warning: the priors do not sum to 1. Renormalizing'
                self.priors = self.priors / self.priors.sum()
                
                
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        params = {"gamma": self.gamma,
                  "degree": self.degree,
                  "coef0": self.coef0}
        try:
            return pairwise_kernels(X, Y, metric=self.kernel,
                                    filter_params=True, **params)
        except AttributeError:
            raise ValueError("%s is not a valid kernel. Valid kernels are: "
                             "rbf, poly, sigmoid, linear and precomputed."
                             % self.kernel)


    def fit(self, X, y):
        """
        Fit the Kernelized Fisher Discriminant model according to the given training data and parameters.
        Based on "Algorithm 5" in
        Zhang, et. al. 'Regularized Discriminant Analysis, Ridge Regression and Beyond' Journal of Machine Learning Research 11 (2010) 2199-2228
        NOTE: setting norm_covariance=False and use_total_scatter=True, and solution_norm = 'A' or 'B' will give the algorithm from paper

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array, shape = [n_samples]
            Target values (integers)
        
        """
        # X, y = check_array(X, y, sparse_format='dense')
        # X, y = check_array(X), check_array(y)#, sparse_format='dense')
        X, y = check_array(X), check_array(y)#, sparse_format='dense')
        self.classes_, y = unique( y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        n_samples_perclass = np.bincount(y)
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        ts = time.time()
                    
        self.means_ = []
        for ind in xrange(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            self.means_.append(np.asarray(meang))
        if self.print_timing: print 'KernelFisher.fit: means took', time.time() - ts


        ts = time.time()
        PI_diag = np.diag( 1.0*n_samples_perclass )                                        # shape(PI_diag) = n_classes x n_classes
        PI_inv = np.diag( 1.0 / (1.0*n_samples_perclass) )                                 # shape(PI_inv) = n_classes x n_classes
        PI_sqrt_inv = np.sqrt( PI_inv )                                                    # shape(PI_sqrt_inv) = n_classes x n_classes
        #H = np.identity(n_samples) - (1.0/(1.0*n_samples))*np.ones((n_samples,n_samples))
        E=np.zeros( (n_samples,n_classes) )                                                # shape(E) = n_samples x n_classes
        E[[range(n_samples),y]]=1
        E_PIsi = np.dot(E, PI_sqrt_inv)
        One_minus_E_Pi_Et = np.identity(n_samples) - np.inner( E, np.inner(PI_diag, E).T ) # shape(One_minus_E_Pi_Et) = n_samples x n_samples
        if self.print_timing: print 'KernelFisher.fit: matrices took', time.time() - ts


        #####################################################################################################################
        #C = HKH = (I - 1/n 1x1.T) K (I - 1/n 1x1.T) = (K -  1xK_mean.T) * (I - 1/n 1x1.T)
        #        = K - K_meanx1.T - 1xK_mean.T + K_allmean 1x1
        #  --> which is the same as what self._centerer.fit_transform(C) performs
        #
        # if use_total_scatter=False,
        #      then using Sw which is (1-E*Pi*E.T)K(1-E*Pi*E.T)
        #####################################################################################################################
        ts = time.time()
        C = self._get_kernel(X) 
        K_mean = np.sum(C, axis=1) / (1.0*C.shape[1])

        if self.use_total_scatter:
            C = self._centerer.fit_transform(C)
        else:
            C = np.inner( One_minus_E_Pi_Et, np.inner(C, One_minus_E_Pi_Et).T)
        if self.print_timing: print 'KernelFisher.fit: Kernel Calculation took', time.time() - ts


        ts = time.time()
        Uc, Sc, Utc, Sc_norm = self.condensed_svd( C, self.tol, store_singular_vals=True )
        if self.print_timing: print 'KernelFisher.fit: Uc, Sc, Utc took', time.time() - ts


        ts = time.time()
        #scale up sigma to appropriate range of singular values
        reg_factor = self.sigma_sqrd * Sc_norm 
        St_reg_inv = np.inner( Uc, np.inner(np.diag(1.0/(Sc + reg_factor)), Utc.T).T )   
        if self.print_timing: print 'KernelFisher.fit: St_reg_inv took', time.time() - ts

        ts = time.time()
        R = np.inner(E_PIsi.T, np.inner(C, np.inner( St_reg_inv, E_PIsi.T ).T ).T )
        if self.print_timing: print 'KernelFisher.fit: R took', time.time() - ts


        ts = time.time()
        Vr, Lr, Vtr, Lr_norm =  self.condensed_svd( R, tol=1e-6 )                
        if self.print_timing: print 'KernelFisher.fit: Vr, Lr, Vtr took', time.time() - ts


        ts = time.time()
        #####################################################################################################################
        #This capital Z is Upsilon.T * H from equation (22)
        #####################################################################################################################
        #Z = np.inner( np.diag(1.0 / np.sqrt(Lr)), np.inner(Vtr, np.inner(E_PIsi.T, np.inner(C, St_reg_inv.T ).T ).T ).T )
        Z = np.inner( np.inner( np.inner( np.inner( np.diag(1.0 / np.sqrt(Lr)), Vtr.T), E_PIsi), C.T), St_reg_inv)

        Z = (Z.T - (Z.sum(axis=1) / (1.0*Z.shape[1])) ).T
        if self.print_timing: print 'KernelFisher.fit: Z took', time.time() - ts

        self.Z = Z
        self.n_components_found_ = Z.shape[0]

        #####################################################################################################################
        #This K_mean is (1/n) K*1_n from equation (22)
        #####################################################################################################################
        self.K_mean = K_mean

        #print Z.shape, K_mean.shape, self.n_components_found_

        self.X_fit_ = X
        return self

    def condensed_svd(self, M, tol=1e-3, store_singular_vals=False):
        U, S, Vt = linalg.svd(M, full_matrices=False)
        if store_singular_vals:
            self.singular_vals = S

        #want tolerance on fraction of variance in singular value
        #when not norm_covariance, need to normalize singular values
        S_norm = np.sum(S)

        rank = np.sum( (S/S_norm) > tol )

        return U[:,:rank], S[:rank], Vt[:rank,:], S_norm


    @property
    def classes(self):
        warnings.warn("KernelFisher.classes is deprecated and will be removed in 0.14. "
                      "Use .classes_ instead.", DeprecationWarning,
                      stacklevel=2)
        return self.classes_

    def _decision_function(self, X):
        #X = np.asarray(X)
        return self.transform(X)

    def decision_function(self, X):
        """
        This function return the decision function values related to each
        class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components_found_]
            Decision function values related to each class, per sample
            n_components_found_ is the number of components requested and found
            NOTE: currently identical to self.transform(X)
        """
        return self._decision_function(X)

    def transform(self, X):
        """
        Project the data so as to maximize class separation (large separation
        between projected class means and small variance within each class).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array, shape = [n_samples, n_components_found_]
        """

        #X = np.asarray(X)
        #ts = time.time()
        k = self._get_kernel(X, self.X_fit_)
        #if self.print_timing: print 'KernelFisher.transform: k took', time.time() - ts

        #ts = time.time()
        z = np.inner(self.Z, (k-self.K_mean) ).T
        #if self.print_timing: print 'KernelFisher.transform: z took', time.time() - ts

        return z
        
    

    def fit_transform(self, X, y, use_total_scatter=True, sigma_sqrd=1e-8, tol=1.0e-3):
        """
        Fit the Fisher Discriminant model according to the given training data and parameters.
        The project the data onto up to n_components_found_ so as to maximize class separation (large separation
        between projected class means and small variance within each class).
        NOTE this function is not clever, it simply runs fit(X,y [, ...]).transform(X)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array, shape = [n_samples]
            Target values (integers)
        store_covariance : boolean
            If True the covariance matrix of each class and each iteration is computed
            and stored in `self.covs_` attribute. has dimensions [n_iterations][2] where 2 is for nclasses = 2

        Returns
        -------
        X_new : array, shape = [n_samples, n_components_found_]
        """
        return self.fit(X, y, use_total_scatter=use_total_scatter, sigma_sqrd=sigma_sqrd, tol=tol).transform(X)
