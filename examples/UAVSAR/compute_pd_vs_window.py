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

# import path of root repo
current_dir = os.path.dirname(os.path.abspath(__file__))
temp = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(1, temp)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
sns.set_style('darkgrid')

from tqdm import tqdm

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
    # DEBUG mode for fast debugging (use a small patch)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    FULL_TIME_SERIES = False # if true: use the full time series, else: use only the first and last images of the time series

    # Activate latex in figures (or not)
    LATEX_IN_FIGURES = True
    if LATEX_IN_FIGURES:
      enable_latex_infigures()

    # Enable parallel processing (or not)
    ENABLE_MULTI = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 6
    number_of_threads_columns = 8
    if os.cpu_count()!=(number_of_threads_rows*number_of_threads_columns):
        print('WARNING: not all cpu are used')

    # data
    image, ground_truth_original, X, Y = load_UAVSAR(PATH, DEBUG, FULL_TIME_SERIES)

    # Parameters
    if DEBUG:
        WINDOWS_SIZES = np.arange(start=5, stop=9, step=2)
        MAXIMUM_RANK = 2
    else:
        WINDOWS_SIZES = np.arange(start=5, stop=17, step=2)
        MAXIMUM_RANK = 5
    PFA_THRESHOLD = 0.1

    n_r, n_rc, p, T = image.shape
    function_to_compute = compute_several_statistics

    # LRCG
    rank_list = [(i+1) for i in range(MAXIMUM_RANK)]
    statistic_list = [scale_and_shape_equality_robust_statistic_low_rank for i in range(len(rank_list))]
    statistic_names = ['$\hat{\Lambda}_{\mathrm{LRCG, R='+str(rank)+'}}$' for rank in rank_list]
    args_list = [(0.01, 20, rank, False, 'log') for rank in rank_list]

    number_of_statistics = len(statistic_list)
    function_args = [statistic_list, args_list]

    # Computing statistics on both images
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   COMPUTING   |') 
    print( '|   in progress |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    t_beginning = time.time()
    results = list()
    m_r_max, m_c_max = None, None
    for windows_size in WINDOWS_SIZES[::-1]:
        windows_mask = np.ones((windows_size,windows_size))
        if (m_r_max is None) and (m_c_max is None):
            m_r_max, m_c_max = windows_mask.shape
        m_r, m_c = windows_mask.shape
        temp = sliding_windows_treatment_image_time_series_parallel(
            image,
            windows_mask,
            function_to_compute,
            function_args,
            multi=ENABLE_MULTI,
            number_of_threads_rows=number_of_threads_rows,
            number_of_threads_columns=number_of_threads_columns)
        diff_m_r = int(m_r_max/2)-int(m_r/2)
        diff_m_c = int(m_c_max/2)-int(m_c/2)
        if (diff_m_r!=0) and (diff_m_c!=0):
            temp = temp[diff_m_r:-diff_m_r, diff_m_c:-diff_m_c]
        results.append(temp)

    # Results: tensor of size (nb_rows, nb_columns, nb_rank_tested, nb_windows_sizes_tested)
    results = np.flip(np.stack(results, axis=-1), axis=-1)

    ground_truth = ground_truth_original[int(m_r_max/2):-int(m_r_max/2), int(m_c_max/2):-int(m_c_max/2)]

    print("Elpased time: %d s" %(time.time()-t_beginning))
    print('Done')

    # Computing pd
    λ_pfa_threshold = np.zeros((len(function_args[0]), len(WINDOWS_SIZES)))
    pd_array = np.zeros((len(function_args[0]), len(WINDOWS_SIZES)))

    for i_w in range(len(WINDOWS_SIZES)):
        for i_s in range(len(statistic_names)):
            # Sorting values of statistic
            λ_vec = np.sort(vec(results[:,:,i_s,i_w]), axis=0)
            λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]
            λ_vec = np.flip(λ_vec, axis=0)

            # Thresholding and summing for each value
            for i_λ, λ in enumerate(λ_vec):
                good_detection = (results[:,:,i_s,i_w] >= λ) * ground_truth
                false_alarms = (results[:,:,i_s,i_w] >= λ) * np.logical_not(ground_truth)
                pfa = false_alarms.sum() / (ground_truth==0).sum()
                if pfa > PFA_THRESHOLD:
                    pd_array[i_s, i_w] = good_detection.sum() / (ground_truth==1).sum()
                    break

    # pd_array is a matrix of shape (nb_rank_tested, nb_windows_sizes_tested)

    # Showing statistics results ROC
    fig = plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    for i_s, statistic in enumerate(statistic_names):
        plt.plot(WINDOWS_SIZES, pd_array[i_s,:], linestyle='--', label=statistic, markersize=4, linewidth=1)
    ax = fig.axes[0]
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel(r'Window size')
    plt.ylabel(r'$P_D$')
    plt.legend()
    plt.show()