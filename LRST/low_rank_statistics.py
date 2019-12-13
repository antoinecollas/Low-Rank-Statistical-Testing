##############################################################################
# Functions to compute Low-Rank statistics
# Authored by Ammar Mian, 10/01/2019
# e-mail: ammar.mian@centralesupelec.fr
##############################################################################
# Copyright 2018 @CentraleSupelec
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
import sys
import numpy as np
import scipy as sp
from sklearn.decomposition._pca import _assess_dimension_

import warnings

from LRST.generic_functions import *


# ----------------------------------------------------------------------------
# 1) Low-Rank Gaussian CM equality statistic
# ----------------------------------------------------------------------------

def Gaussian_Log_Likelihood(𝐗, 𝛍, 𝚺):
    """ Gaussian Log-Likelihood of i.i.d observations
        ----------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * 𝛍 = mean (array of dimension p)
            * 𝚺 = covariance matrix of dimension (p,p)
        Outputs:
        ---------
            * the Log-Likelihood """
    
    (p, N) = 𝐗.shape
    𝐗_centered = 𝐗 - np.tile(𝛍.reshape((p,1)), (1,N))
    𝐒 = SCM(𝐗_centered)
    ℓogℒ = - N*p*np.log(np.pi) - N*np.log(np.abs(np.linalg.det(𝚺))) - \
            np.trace(𝐒@np.linalg.inv(𝚺))
    return ℓogℒ

def LR_𝜮(𝚺, R,  σ2):
    """ Low-Rank operator on 𝚺
        ----------------------------------------------
        Inputs:
        --------
            * 𝚺 = covariance matrix of dimension (p,p)
            * R = rank
            * σ2 = noise level
        Outputs:
        ---------
            * the LR regularised matrix """

    (p,p) = 𝚺.shape
    if R==p:
        return 𝚺

    u, s, vh = np.linalg.svd(𝚺)
    if σ2 is None:
        σ2 = np.mean(s[R:])
    s_signal = np.max([s[:R],σ2*np.ones((R,))], axis=0)
    s_noise = σ2 * np.ones((p-R,))
    s = np.diag(np.hstack([s_signal, s_noise]))
    𝚺_R = u @ s @ u.conj().T

    return 𝚺_R


def information_criterion(s):
    """ Implements information criterion of AIC/BIC methods
    ----------------------------------------------
    Inputs:
    --------
        * s = (p) numpy array of eigenvalues
    Outputs:
    ---------
        * the information criterion """
    p = len(s)
    k = np.arange(1, p)
    numerator = np.empty(len(k))
    for idx in k:
        power = 1/(p-idx)
        temp = np.prod(s[idx:])**power
        numerator[idx-1] = temp
    denominator = np.cumsum(s[1:][::-1])[::-1]
    ic = -np.log((p-k)*numerator/denominator)
    return ic

def AIC_criterion(s, n):
    """ AIC criterion for order selection
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    k = np.arange(1,p)
    ic = information_criterion(s)
    criterion = n*(p-k)*ic+k*(2*p-k)
    return criterion
    
def AICc_criterion(s, n):
    """ AICc criterion for order selection
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    k = np.arange(1,p)
    ic = information_criterion(s)
    nb_param = k*(2*p-k) + 1
    criterion = n*(p-k)*ic + nb_param + (nb_param**2+nb_param)/((n*p)-nb_param-1)
    return criterion

def BIC_criterion(s, n):
    """ BIC criterion for order selection
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    k = np.arange(1,p)
    ic = information_criterion(s)
    criterion = n*(p-k)*ic+k*(2*p-k)*0.5*np.log(n)
    return criterion

def BIC_minka_criterion(s, n):
    """ BIC criterion for order selection from Minka 2000 "Automatic choice of dimensionality for PCA"
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    k = np.arange(1, p)
    criterion = np.empty(p-1)
    for idx in k:
        m = p*(p-1)/2-(p-idx)*(p-idx-1)/2
        temp0 = (n/2) * np.sum(np.log(s[:idx]))
        temp1 = (n*(p-idx)/2) * np.log(np.sum(s[idx:])/(p-idx))
        criterion[idx-1] = temp0 + temp1 + ((m+idx)/2)*np.log(n)
    return criterion

def Minka_criterion(s, n):
    """ Minka criterion for order selection from Minka 2000 "Automatic choice of dimensionality for PCA"
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    criterion = np.empty(p)
    for rank in range(p):
        criterion[rank] = -_assess_dimension_(s, rank, n, p)
    return criterion

def EDC_criterion(s, n):
    """ EDC criterion for order selection
    ----------------------------------------------
    Inputs:
    --------
        * s = a (p) numpy array of eigenvalues of SCM
        * n = number of samples used for computing the SCM
    Outputs:
    ---------
        * the criterion """
    p = len(s)
    k = np.arange(1,p)
    ic = information_criterion(s)
    criterion = n*(p-k)*ic+k*(2*p-k)*np.sqrt(n*np.log(np.log(n)))
    return criterion

def SCM_rank_criterion(𝐗, method):
    """ Compute the SCM of 𝐗 and the AIC/BIC criteria for rank estimation
    ----------------------------------------------
    Inputs:
    --------
        * 𝐗 = a (p, N) numpy array with:
            * p = dimension of vectors
            * N = number of Samples
        * method = 'AIC'/'AICc'/'BIC'/'BIC_Minka'/'Minka'/'EDC' and their thresholded versions
    Outputs:
    ---------
        * the criterion """
    (p, N) = 𝐗.shape
    𝚺 = SCM(X)
    u, s, vh = np.linalg.svd(𝚺)
    if method == 'AIC':
        criterion = AIC_criterion(s, N)
    elif method == 'AICc':
        criterion = AICc_criterion(s, N)
    elif method == 'BIC':
        criterion = BIC_criterion(s, N)
    elif method == 'BIC_Minka':
        criterion = BIC_minka_criterion(s, N)
    elif method == 'Minka':
        criterion = Minka_criterion(s, N)
    elif method == 'EDC':
        criterion = EDC_criterion(s, N)
    else:
        print('Method', method, 'unknown...')
        sys.exit(1)
    return criterion

def rank_estimation(𝐗, method='AIC'):
    """ order selection using AIC or BIC methods
        ----------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
        Outputs:
        ---------
            * the Rank """
    method_split = method.split('_')
    if (len(method_split)>=4) or (len(method_split)==2) or ((len(method_split)==3) and (method_split[1]!='thresholded')):
        print('Method', method, 'unknown...')
        sys.exit(1)
    method = method_split[0]
    if len(method_split) == 3:
        threshold = int(method_split[2])
    else:
        threshold = None
    criterion = SCM_rank_criterion(𝐗, method)
    rank = np.argmin(criterion) + 1
    if (threshold is not None) and (rank>threshold):
        rank = threshold
    return rank

def rank_estimation_reshape(𝐗, args):
    """ Estimates rank using AIC rule
    Input:
        * 𝐗 = a tensor of size (p, ...) with each observation along column dimension
        * args = method to use (either AIC or BIC)
    Outputs:
        * R = rank
    """
    p = 𝐗.shape[0]
    𝐗 = 𝐗.reshape((p, -1))
    R = rank_estimation(𝐗, method=args)
    return R

def eigenvalues_SCM(𝐗, args):
    """ compute SCM and its eigenvalues
    Input:
        * 𝐗 = a tensor of size (p, ...) with each observation along column dimension
        * args = Nothing
    Outputs:
        * eigenvalues
    """
    p = 𝐗.shape[0]
    𝐗 = 𝐗.reshape((p, -1))
    𝚺 = SCM(𝐗)
    s = np.linalg.eigvals(𝚺)
    s = np.sort(s)[::-1]
    return s

def σ2_estimation(𝐗, R):
    """ Estimate σ2 locally using samples
        ----------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
            * R = rank
        Outputs:
        ---------
            * σ2 """

    (p, N) = 𝐗.shape
    𝚺 = SCM(𝐗)
    u, s, vh = np.linalg.svd(𝚺)
    if R<p:
        σ2 = s[R:].mean()
    else:
        σ2 = 0
    return σ2


def LR_CM_equality_test(𝐗, args):
    """ Gaussian Low-Rank covariance matrix equality GLRT
        --------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
                * T = Number of groups of samples
            * args = (R, σ2, scale) with
                * R = rank (put AIC or BIC for adaptive estimation)
                * σ2 assumed known (boolean) = if true we estimate σ2 at the beginning, 
                else it is taken as the mean of p-R lowest eigenvalues when needed.
                * scale = 'linear' or 'log'
        Outputs:
        ---------
            * the GLRT """

    # 1) Useful variables
    (p, N, T) = 𝐗.shape
    R, σ2, scale = args

    # 2) Estimate R and σ2 if needed
    if not (isinstance(R, int)) :
        R = rank_estimation(𝐗.reshape((p, N*T)), R)
    if σ2:
        σ2 = σ2_estimation(𝐗.reshape((p,N*T)), R)
    else:
        σ2 = None

    # 3) Estimate 𝚺_R under ℋ0 hypothesis
    𝚺 = SCM(X.reshape((p,N*T)))
    𝜮_R = LR_𝜮(𝚺, R, σ2)

    # 4) Estimate 𝚺t_R under ℋ1 hypothesis
    𝜮t_R = np.zeros((p,p,T)).astype(complex)
    for t in range(0, T):
        𝚺t =  SCM(𝐗[:, :, t])
        𝜮t_R[:,:,t] = LR_𝜮(𝜮t, R,  σ2)

    # 5) Compute ratio
    𝛍 = np.zeros(p)
    ℓogℒ_ℋ0 = 0
    ℓogℒ_ℋ1 = 0
    for t in range(0, T):
        ℓogℒ_ℋ0 =  ℓogℒ_ℋ0 + Gaussian_Log_Likelihood(𝐗[:,:,t], 𝛍, 𝜮_R)
        ℓogℒ_ℋ1 =  ℓogℒ_ℋ1 + Gaussian_Log_Likelihood(𝐗[:,:,t], 𝛍, 𝜮t_R[:,:,t])

    if scale=='log':
        return np.real(ℓogℒ_ℋ1 - ℓogℒ_ℋ0)
    else:
        return np.exp(np.real(ℓogℒ_ℋ1 - ℓogℒ_ℋ0))


def LR_Plug_in_CM_equality_test(𝐗, args):
    """ Gaussian Plug-in Low-Rank covariance matrix equality GLRT
        ------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
                * T = Number of groups of samples
            * args = (R, σ2, scale) with
                * R = rank (put 0 for adaptive estimation)
                * σ2 assumed known (boolean) = if true we estimate σ2 at the beginning, 
                else it is taken as the mean of p-R lowest eigenvalues when needed.
                * scale = 'linear' or 'log'
        Outputs:
        ---------
            * the plug-in GLRT """

    # 1) Useful variables
    (p, N, T) = 𝐗.shape
    R, σ2, scale = args

    # 2) Estimate R and σ2 if needed
    if not (isinstance(R, int)) :
        R = rank_estimation(𝐗.reshape((p, N*T)), R)
    if σ2:
        σ2 = σ2_estimation(𝐗.reshape((p,N*T)), R)
    else:
        σ2 = None

    S = SCM(𝐗.reshape((p, N*T)))
    S = LR_𝜮(S, R,  σ2)
    logNumerator = N * T * np.log(np.abs(np.linalg.det(S)))
    logDenominator = 0
    for t in range(0, T):
        St = SCM(𝐗[:, :, t])
        St = LR_𝜮(St, R,  σ2)
        logDenominator = logDenominator + N * np.log(np.abs(np.linalg.det(St)))
    

    if scale=='log':
        return np.real(logNumerator - logDenominator)
    else:
        return np.exp(np.real(logNumerator - logDenominator))

# ----------------------------------------------------------------------------
# 2) Low-Rank SIRV CM equality statistic
# ----------------------------------------------------------------------------

def tyler_estimator_covariance_low_rank(𝐗, R, σ2, tol=0.001, iter_max=20):
    """ A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * tol = tolerance for convergence of estimator
            * R = Rank
            * σ2 = noise level
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation
    (p,N) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = SCM(𝐗) # Initialise estimate to identity
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and (iteration<iter_max):
        
        # Computing expression of Tyler estimator (with matrix multiplication)
        τ = np.diagonal(𝐗.conj().T@np.linalg.inv(𝚺)@𝐗)
        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing low rank structure
        𝚺_new = LR_𝜮(𝚺_new, R, σ2)

        # Normalisation
        𝚺_new = 𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    return (𝚺, δ, iteration)

def tyler_estimator_covariance_matandtext_low_rank(𝐗, R, σ2, tol, iter_max):
    """ A function that computes the Modified Tyler Fixed Point Estimator Low Rank for 
    covariance matrix estimation under problem MatAndText.
        Inputs:
            * 𝐗 = a matrix of size p*N*T with each saptial observation along column dimension and time
                observation along third dimension.
            * R = Rank
            * σ2 = noise lvel
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
        Outputs:
            * 𝚺 = the estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """
    (p, N, T) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = SCM(𝐗.reshape((p,T*N))) # Initialise estimate to SCM
    iteration = 0

    # Recursive algorithm
    while (δ>tol) and iteration < iter_max:

        # Compute the textures for each pixel using all the dates available
        τ = 0
        i𝚺 = np.linalg.inv(𝚺)
        for t in range(0, T):
            τ = τ + np.diagonal(𝐗[:,:,t].conj().T@i𝚺@𝐗[:,:,t])

        # Computing expression of the estimator
        𝚺_new = 0
        for t in range(0, T):
            𝐗_bis = 𝐗[:,:,t] / np.sqrt(τ)
            𝚺_new = 𝚺_new + (p/N) * 𝐗_bis@𝐗_bis.conj().T

        # Imposing low rank structure
        𝚺_new = LR_𝜮(𝚺_new, R, σ2)

        # Normalisation
        𝚺_new = 𝚺_new/np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')

        # Updating 𝚺
        𝚺 = 𝚺_new
        iteration = iteration + 1


    return (𝚺, δ, iteration)

def scale_and_shape_equality_robust_statistic_low_rank(𝐗, args):
    """ Low-Rank GLRT test for testing a change in the scale or/and shape of 
        a deterministic SIRV model.
        Inputs:
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples at each date
                * T = length of time series
            * args = (tol, iter_max, R, σ2, scale)
                * tol = tol for tyler estimation
                * iter_max = maximum number of iterations for tyler estimation
                * R = rank (put AIC or BIC for adaptive estimation)
                * σ2 assumed known (boolean) = if true we estimate σ2 at the beginning, 
                else it is taken as the mean of p-R lowest eigenvalues when needed.
                * scale = 'linear' or 'log'
        Outputs:
            * the statistic given the observations in input"""


    tol, iter_max, R, σ2, scale = args
    (p, N, T) = 𝐗.shape

    # Estimate R and σ2 if needed
    if not (isinstance(R, int)) :
        R = rank_estimation(𝐗.reshape((p, N*T)), R)
    if σ2:
        σ2 = σ2_estimation(𝐗.reshape((p,N*T)), R)
    else:
        σ2 = None

    # Estimating 𝚺_0 using all the observations
    (𝚺_0, δ, niter) = tyler_estimator_covariance_matandtext_low_rank(𝐗, R, σ2, tol, iter_max)
    i𝚺_0 = np.linalg.inv(𝚺_0)

    # Some initialisation
    log_numerator_determinant_terms = T*N*np.log(np.abs(np.linalg.det(𝚺_0)))
    log_denominator_determinant_terms = 0
    𝛕_0 = 0
    log𝛕_t = 0

    # Iterating on each date to compute the needed terms
    for t in range(0,T):
        # Estimating 𝚺_t
        (𝚺_t, δ, iteration) = tyler_estimator_covariance_low_rank(𝐗[:,:,t], R, σ2, tol, iter_max)

        # Computing determinant add adding it to log_denominator_determinant_terms
        log_denominator_determinant_terms = log_denominator_determinant_terms + \
                                            N*np.log(np.abs(np.linalg.det(𝚺_t)))

        # Computing texture estimation
        𝛕_0 =  𝛕_0 + np.diagonal(𝐗[:,:,t].conj().T@i𝚺_0@𝐗[:,:,t]) / T
        log𝛕_t = log𝛕_t + np.log(np.diagonal(𝐗[:,:,t].conj().T@np.linalg.inv(𝚺_t)@𝐗[:,:,t]))

    # Computing quadratic terms
    log_numerator_quadtratic_terms = T*p*np.sum(np.log(𝛕_0))
    log_denominator_quadtratic_terms = p*np.sum(log𝛕_t)

    # Final expression of the statistic
    if scale=='linear':
        λ = np.exp(np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms))
    else:
        λ = np.real(log_numerator_determinant_terms - log_denominator_determinant_terms + \
        log_numerator_quadtratic_terms - log_denominator_quadtratic_terms)

    return λ