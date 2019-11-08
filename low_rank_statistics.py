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
import numpy as np
import scipy as sp
import warnings
from generic_functions import *




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
            np.trace(𝐒@np.linalg.inv(𝜮))
    return ℓogℒ

def LR_𝜮(𝜮, R,  σ2):
    """ Low-Rank operator on 𝜮
        ----------------------------------------------
        Inputs:
        --------
            * 𝚺 = covariance matrix of dimension (p,p)
            * R = rank
            * σ2 = noise level
        Outputs:
        ---------
            * the LR regularised matrix """

    (p,p) = 𝜮.shape
    u, s, vh = np.linalg.svd(𝜮)
    s_signal = np.max([s[:R],σ2*np.ones((R,))], axis=0)
    s_noise = σ2 * np.ones((p-R,))
    s = np.diag(np.hstack([s_signal, s_noise]))
    𝚺_R = u @ s @ u.conj().T

    return 𝚺_R


def Rank_estimation(𝐗):
    """ Not implemented yet but model order selection
        ----------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
        Outputs:
        ---------
            * the Rank """

    (p, N) = 𝐗.shape
    return p 


def σ2_estimaton(𝐗, R):
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
        σ2 = s[R+1:].mean()
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
                * R = rank (put 0 for adaptive estimation)
                * σ2 = noise level (put 0 for adaptive estimation)
                * scale = 'linear' or 'log'
        Outputs:
        ---------
            * the GLRT """

    # 1) Useful variables
    (p, N, T) = 𝐗.shape
    R, σ2, scale = args

    # 2) Estimate R and σ2 if needed
    if not R:
        R = Rank_estimation(𝐗.reshape((p, N*T)))
    if not σ2:
        σ2 = σ2_estimaton(𝐗.reshape((p,N*T)), R)

    # 3) Estimate 𝚺_R under ℋ0 hypothesis
    𝜮 = SCM(X.reshape((p,N*T)))
    𝜮_R = LR_𝜮(𝜮, R, σ2)

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
                * σ2 = noise level (put 0 for adaptive estimation)
                * scale = 'linear' or 'log'
        Outputs:
        ---------
            * the plug-in GLRT """

    # 1) Useful variables
    (p, N, T) = 𝐗.shape
    R, σ2, scale = args

    # 2) Estimate R and σ2 if needed
    if not R:
        R = Rank_estimation(𝐗.reshape((p, N*T)))
    if not σ2:
        σ2 = σ2_estimaton(𝐗.reshape((p,N*T)), R)

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
