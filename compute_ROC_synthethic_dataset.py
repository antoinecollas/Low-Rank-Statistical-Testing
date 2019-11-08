##############################################################################
# Simulation to test LR CM equality statistic on synthetic data
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
from generic_functions import *
import matplotlib.pyplot as plt
from monte_carlo_tools import *
from change_detection_functions import *
from low_rank_statistics import *
from proportionality_statistics import *
import os
import time
import scipy as sp
import seaborn as sns
from tqdm import tqdm
sns.set_style("darkgrid")

if __name__ == '__main__':

	# ---------------------------------------------------------------------------------------------------------------
    # Python setting
    # ---------------------------------------------------------------------------------------------------------------

    # Activate latex in figures (or not)
    latex_in_figures = True
    if latex_in_figures:
      enable_latex_infigures()

    # Enable parallel processing (or not)
    enable_multi = True
    # These two variables serves to split an image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 8
    number_of_threads_columns = 6
    number_of_threads = number_of_threads_rows * number_of_threads_columns

	# ---------------------------------------------------------------------------------------------------------------
    # Simulation Parameters
    # ---------------------------------------------------------------------------------------------------------------
    p = 20 # Dimension of data
    N = 25 # Number of samples
    T = 2 # Number of groups of samples
    R = 5 # Rank of signal
    SNR_dB = 10 # SNR of the signal 
    σ2 = 1 # Noise level
    τ = σ2*10**(SNR_dB/20) # Signal level
    𝚺_0 = τ*ToeplitzMatrix(0.1, p) + σ2*np.eye(p) # Covariance matrix no change
    𝚺_1 = τ*ToeplitzMatrix(0.7, p) + σ2*np.eye(p) # Covariance matrix if change
    number_of_trials = 9600 # Number of MonteCarlo simulations
    number_of_points = 50 # Number of points for plotting

    # ---------------------------------------------------------------------------------------------------------------
    # Setting the statistics for the simulation
    # ---------------------------------------------------------------------------------------------------------------
    function_to_compute = compute_several_statistics
    statistics_list = [covariance_equality_glrt_gaussian_statistic, LR_CM_equality_test, proportionality_statistic_marginal]
    statistics_names = [r'$\hat{\Lambda}_{\mathrm{Eq}}$', r'$\hat{\Lambda}_{\mathrm{lrEq}}$', r'$\hat{\Lambda}_{\mathrm{P}}$']
    args_list = ['log', (R, σ2, 'log'), (20,0.01,'log')]
    number_of_statistics = len(statistics_list)
    function_args = [statistics_list, args_list]

    # ---------------------------------------------------------------------------------------------------------------
    # Select zero-mean, no pseudo-covariance, Gaussian distribution for data generation
    # ---------------------------------------------------------------------------------------------------------------
    data_generation_function = generate_time_series_multivariate_vector
    generation_function = wrapper_multivariate_complex_normal_samples
    𝛍 = np.zeros(p)
    pseudo_𝚺 = 0

	# ---------------------------------------------------------------------------------------------------------------
    # Pfa Simulation
    # ---------------------------------------------------------------------------------------------------------------
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   Pfa MC      |') 
    print( '|   simulation  |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()

    # Generating parameters to pass to the Monte-Carlo function: since
    # the series is homogeneous, we repeat T times
    data_args_list = [[𝛍, 𝚺_0, N, pseudo_𝚺]]*T
    data_generation_args = [p, N, T, generation_function, data_args_list]
    # Doing simulation
    λ_vec_temp = np.array(compute_monte_carlo_parallel(data_generation_function, data_generation_args, 
                                        function_to_compute, function_args, 
                                        number_of_trials, multi=enable_multi, number_of_threads=number_of_threads))
    # Sorting values
    Pfa_vec = np.arange(number_of_trials, 0,-1)/number_of_trials
    λ_vec_temp = np.sort(λ_vec_temp, axis=0)
    indices_λ = np.unique((np.floor(np.linspace(0, number_of_trials-1, num=number_of_points))).astype(int)) # Indexes for thresholding
    λ_vec = λ_vec_temp[indices_λ]
    Pfa_vec = Pfa_vec[indices_λ]
    print("Elpased time: %d s" %(time.time()-t_beginning))


	# ---------------------------------------------------------------------------------------------------------------
    # PD Simulation
    # ---------------------------------------------------------------------------------------------------------------
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   PD MC       |') 
    print( '|   simulation  |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()

    # Generating parameters to pass to the Monte-Carlo function
    data_args_list = [[𝛍, 𝚺_0, N, pseudo_𝚺], [𝛍, 𝚺_1, N, pseudo_𝚺]]
    data_generation_args = [p, N, T, generation_function, data_args_list]
    # Doing simulation
    results = np.array(compute_monte_carlo_parallel(data_generation_function, data_generation_args, 
                                        function_to_compute, function_args, 
                                        number_of_trials, multi=enable_multi, number_of_threads=number_of_threads))

    # Computing PD for each value of threshold
    PD_vec = np.zeros((len(Pfa_vec), number_of_statistics))
    for i_λ, λ in enumerate(λ_vec):
    	PD_vec[i_λ, :] = np.sum(results >= λ, axis=0)/number_of_trials
    print("Elpased time: %d s" %(time.time()-t_beginning))

	# ---------------------------------------------------------------------------------------------------------------
    # Plotting ROC
    # ---------------------------------------------------------------------------------------------------------------
    markers = ['o', 's', 'd', '*', '+']
    plt.figure(figsize=(6, 4), dpi=80, facecolor='w')
    for i_s, statistic in enumerate(statistics_names):
        plt.plot(Pfa_vec, PD_vec[:,i_s], linestyle='--', label=statistic,
            marker=markers[i_s], fillstyle='none')
    plt.title('ROC curves')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.legend()
    plt.show()
    