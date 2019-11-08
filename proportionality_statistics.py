##############################################################################
# Functions to compute proportionality testing statistics
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
# 1) Expression of the Generalised Fixed Point Estimate (GFPE)
# ----------------------------------------------------------------------------
def GFPE(𝐗, 𝐊_vec, iter_max=20, ϵ=0.001):
    """ Implementation of the Generalised Fixed Point Estimate (GFPE)
        A. Taylor, P. Forster, F. Daout, H. M. Oriot and L. Savy, 
        "A Generalization of the Fixed Point Estimate for Packet-Scaled Complex Covariance Matrix Estimation," 
        in IEEE Transactions on Signal Processing, vol. 65, no. 20, pp. 5393-5405, 15 Oct.15, 2017.
        doi: 10.1109/TSP.2017.2731324
        --------------------------------------------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = max number of samples for each group
            * 𝐊_vec = [K_1, ..., K_T] with
                * K_i = number of samples for group i
            * iter_max = number of iteratins max of the algorithm
            * ϵ = value for convergence criterion
        Outputs:
        ---------
            * (𝜮, 𝛃, n_iter, δ) with
                * 𝜮 = the GFPE
                * 𝛃 = [β_1, ..., β_T] are the estimates of proportionality coeffs
                * n_iter = number of iterations done
                * δ = value of convergence criterion """ 

    # Initialisation
    (p,N,T) = 𝐗.shape
    δ = np.inf # Distance between two iterations
    𝚺 = np.eye(p) # Initialise estimate to identity
    𝛃 = np.zeros(T)
    K = np.sum(𝐊_vec)
    iteration = 0

    # To save time, we compute each SCM now
    𝓢 = np.zeros((p,p, T)).astype(complex)  # Contains all SCMS
    for i in range(0,T):
        𝓢[:,:,i] = SCM(𝐗[:,:,i])

    # Recursive algorithm
    while (δ>ϵ) and (iteration<iter_max):

        # Computing estimates of β_i and accumulating sum for 𝚺
        𝚺_new = 0
        i𝚺 = np.linalg.inv(𝚺)
        for i in range(T):
            𝛃[i] = np.real(np.trace( 𝓢[:,:,i] @ i𝚺 )/p)
            𝚺_new = 𝚺_new + (𝐊_vec[i]/𝛃[i]) * 𝓢[:,:,i]

        # Nomalising 𝚺_new by the trace
        𝚺_new = 𝚺_new / K
        𝚺_new = p * 𝚺_new/np.trace(𝚺_new)

        # Computing convergence criterions
        δ = np.linalg.norm(𝚺_new - 𝚺, 'fro') / np.linalg.norm(𝚺, 'fro')
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new

    return (𝚺_new, 𝛃, iteration, δ)


# ----------------------------------------------------------------------------
# 2) Proportionality test statistic marginal (one versus many groups)
# ----------------------------------------------------------------------------
def proportionality_statistic_marginal(𝐗, args):
    """ Implementation of the proportionality testing of:
        A. Taylor, H. Oriot, P. Forster, F. Brigui, L. Savy and F. Daout, 
        "Reducing false alarm rate by testing proportionality of covariance matrices,"
         International Conference on Radar Systems (Radar 2017), Belfast, 2017, pp. 1-4.
         doi: 10.1049/cp.2017.0405
         WARNNG: number of samples for each gorup is the same here for convenience purpose !
        --------------------------------------------------------------------------------------------------------
        Inputs:
        --------
            * 𝐗 = a (p, N, T) numpy array with:
                * p = dimension of vectors
                * N = number of Samples
                * T = Number of groups of samples
            * args = (iter_max, ϵ, scale) with
                * iter_max = number of iterations max for GFPE
                * ϵ = tolerance for GFPE
                * scale = 'linear' or 'log'
        Outputs:
        ---------
            * the test statistic """

    # Some useful definitions
    (p,N,T) = 𝐗.shape
    iter_max, ϵ, scale = args

    # Estimating parameters under ℋ0 hypothesis
    𝚺gfp_ℋ0, 𝛃_ℋ0, iteration, δ = GFPE(𝐗, [N]*T, iter_max, ϵ)

    # Estimating parameters under ℋ1 hypothesis
    𝚺gfp_ℋ1, 𝛃_ℋ1, iteration, δ = GFPE(𝐗[:,:,:-1], [N]*(T-1), iter_max, ϵ)
    𝚺0_ℋ1 = SCM(𝐗[:,:,0])

    # Computing statistic
    logλ = np.log(np.abs(np.linalg.det(𝛃_ℋ0[0]*𝚺gfp_ℋ0))) - np.log(np.abs(np.linalg.det(𝚺0_ℋ1)))
    for i in range(1,T):
        logλ = logλ + np.log(np.abs(np.linalg.det(𝛃_ℋ0[i]*𝚺gfp_ℋ0))) - \
                        np.log(np.abs(np.linalg.det(𝛃_ℋ1[i-1]*𝚺gfp_ℋ1)))
    logλ = N*logλ

    if scale=='log':
        return np.real(logλ)
    else:
        return np.exp(np.real(logλ))

