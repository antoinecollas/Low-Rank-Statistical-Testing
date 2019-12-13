##############################################################################
# ROC plots on real data UAVSAR
# WARNING: will download about 28 Go of data
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
import os, sys, time

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

from LRST.generic_functions import *
from LRST.monte_carlo_tools import *
from LRST.multivariate_images_tools import *
from LRST.change_detection_functions import *
from LRST.low_rank_statistics import *
from LRST.proportionality_statistics import *
from LRST.read_sar_data import *
from LRST.wavelet_functions import *
from compute_ROC_UAVSAR_dataset import load_UAVSAR

if __name__ == '__main__':
    # Activate latex in figures (or not)
    LATEX_IN_FIGURES = True
    if LATEX_IN_FIGURES:
      enable_latex_infigures()

    # Enable parallel processing (or not)
    ENABLE_MULTI = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 4
    number_of_threads_columns = 4
    if os.cpu_count()!=(number_of_threads_rows*number_of_threads_columns):
        print('WARNING: not all cpu are used')
    
    # data

    # DEBUG mode for fast debugging (use a small patch)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    FULL_TIME_SERIES = False # if true: use the full time series, else: use only the first and last images of the time series
    SCENE_NUMBER = 1

    image, ground_truth_original, X, Y = load_UAVSAR(PATH, DEBUG, FULL_TIME_SERIES, SCENE_NUMBER)

    # Parameters
    n_r, n_rc, p, T = image.shape
    windows_mask = np.ones((5,5))
    m_r, m_c = windows_mask.shape
    function_to_compute = compute_several_statistics

    ground_truth = ground_truth_original[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]

    # Rank estimation
    statistic_list = [rank_estimation_reshape]
    args_list = [('EDC_thresholded_4')]
    function_args = [statistic_list, args_list]

    print( '|￣￣￣￣￣￣￣￣|')
    print( '|     RANK    |') 
    print( '|  estimation |')
    print( '| ＿＿＿＿＿＿＿|')
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()
    ranks = sliding_windows_treatment_image_time_series_parallel(image, windows_mask, function_to_compute, function_args, multi=ENABLE_MULTI, number_of_threads_rows=number_of_threads_rows, number_of_threads_columns=number_of_threads_columns)
    print("Elpased time: %d s" %(time.time()-t_beginning))

    # Test with EDC
    statistic_list = [LR_CM_equality_test, LR_CM_equality_test]
    statistic_names = ['$\hat{\Lambda}_{\mathrm{LRG, R=1}}$', '$\hat{\Lambda}_{\mathrm{LRG, R=EDC}}$']
    args_list = [(1, False, 'log'), ('EDC_thresholded_4', False, 'log')]
    function_args = [statistic_list, args_list]
    if not (len(statistic_list)==len(statistic_names)==len(args_list)):
        print('ERROR')
        sys.exit(1)

    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   COMPUTING   |') 
    print( '|   in progress |')
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()
    results = sliding_windows_treatment_image_time_series_parallel(image, windows_mask, function_to_compute, function_args, multi=ENABLE_MULTI, number_of_threads_rows=number_of_threads_rows, number_of_threads_columns=number_of_threads_columns)
    print("Elpased time: %d s" %(time.time()-t_beginning))

    # Computing ROC curves
    rank_values = np.unique(ranks)
    nb_thresholds = 1000
    nb_points = nb_thresholds//10
    pfa_array = np.zeros((nb_points, len(function_args[0])))
    pd_array = np.zeros((nb_points, len(function_args[0])))

    # Compute ROC curve for non adaptative model
    # Sorting values of statistic
    λ_vec = np.sort(vec(results[:,:,0]), axis=0)
    λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]
    # Selectionning nb_points values from beginning to end
    indices_λ = np.floor(np.linspace(0, len(λ_vec)-1, num=nb_points)) # logspace
    λ_vec = np.flip(λ_vec, axis=0)
    λ_vec = λ_vec[indices_λ.astype(int)]
    # Thresholding and summing for each value
    for i_λ, λ in enumerate(λ_vec):
        good_detection = (results[:,:,0] >= λ) * ground_truth
        false_alarms = (results[:,:,0] >= λ) * np.logical_not(ground_truth)
        pd_array[i_λ, 0] = good_detection.sum() / (ground_truth==1).sum()
        pfa_array[i_λ, 0] = false_alarms.sum() / (ground_truth==0).sum()

    pfa_array_ranks = np.zeros((len(ranks), nb_thresholds))
    pd_array_ranks = np.zeros((len(ranks), nb_thresholds))

    # Compute ROC curve for low rank model
    for rank in rank_values:
        mask = (ranks == rank)[:,:,0]
        results_rank = results[:,:,1][mask]
        ground_truth_rank = ground_truth[mask]

        # Sorting values of statistic
        λ_vec = np.sort(vec(results_rank), axis=0)
        λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]
        # Selectionning nb_thresholds values from beginning to end
        indices_λ = np.floor(np.linspace(0, len(λ_vec)-1, num=nb_thresholds)) # logspace
        λ_vec = np.flip(λ_vec, axis=0)
        λ_vec = λ_vec[indices_λ.astype(int)]
        # Thresholding and summing for each value
        for i_λ, λ in enumerate(λ_vec):
            good_detection = (results_rank >= λ) * ground_truth_rank
            false_alarms = (results_rank >= λ) * np.logical_not(ground_truth_rank)
            pd_array_ranks[rank, i_λ] = good_detection.sum() / (ground_truth_rank==1).sum()
            pfa_array_ranks[rank, i_λ] = false_alarms.sum() / (ground_truth_rank==0).sum()

    pfa_target = np.linspace(1/nb_points, 1, num=nb_points)
    for i, pfa in enumerate(pfa_target):
        pd = 0
        for rank in rank_values:
            j = 0
            while pfa_array_ranks[rank,j]<pfa:
                j += 1
            relative_error = np.abs(pfa-pfa_array_ranks[rank,j])/pfa
            if relative_error>=0.05:
                print('High relative error on Pfa:', relative_error)
                print('rank:', rank)
                print('pfa:', pfa)
                print('pfa_array_ranks[rank, j]:', pfa_array_ranks[rank, j])
                print()
                if not DEBUG:
                    sys.exit(1)
            nb_points_rank = (ranks == rank).sum()
            pd += nb_points_rank*pd_array_ranks[rank, j]
        pfa_array[i, 1] = pfa
        pd_array[i, 1] = pd/(results.shape[0]*results.shape[1])

    # Showing statistics results raw
    for i_s, statistic in enumerate(statistic_names):
        plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
        image_temp = np.nan*np.ones((n_r, n_rc))
        image_temp[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)] = (results[:,:,i_s] - results[:,:,i_s].min()) / (results[:,:,i_s].max() - results[:,:,i_s].min())
        plt.pcolormesh(X,Y, image_temp, cmap='jet')
        plt.xlabel(r'Azimuth (m)')
        plt.ylabel(r'Range (m)')
        plt.title(statistic)
        plt.colorbar()

    # Showing statistics results ROC
    plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    for i_s, statistic in enumerate(statistic_names):
        plt.plot(pfa_array[:,i_s], pd_array[:,i_s], linestyle='--', label=statistic, markersize=4, linewidth=1)
    plt.xlabel(r'$\mathrm{P}_{\mathrm{FA}}$')
    plt.ylabel(r'$\mathrm{P}_{\mathrm{D}}$')
    plt.legend()
    plt.xlim([0.,1])
    plt.ylim([0,1])
    plt.show()