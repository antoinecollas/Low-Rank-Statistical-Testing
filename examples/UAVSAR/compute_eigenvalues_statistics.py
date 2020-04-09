##############################################################################
# Compute eigenvalues statistics
# WARNING: will download about 28 Go of data
# Authored by Ammar Mian, 10/01/2019
# e-mail: ammar.mian@centralesupelec.fr
# Modified by Antoine Collas, 10/2019
# e-mail: antoine.collas@centralesupelec.fr
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
sns.set_style("darkgrid")

from LRST.generic_functions import *
import matplotlib.pyplot as plt
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
    LATEX_IN_FIGURES = False
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

    # DEBUG mode for fast debugging (use a small patch of 200x200 pixels)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    FULL_TIME_SERIES = False # if true: use the full time series, else: use only the first and last images of the time series

    image, ground_truth_original, X, Y = load_UAVSAR(PATH, DEBUG, FULL_TIME_SERIES)

    # Parameters
    n_r, n_rc, p, T = image.shape
    windows_mask = np.ones((5,5))
    m_r, m_c = windows_mask.shape
    function_to_compute = compute_several_statistics

    ground_truth = ground_truth_original[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]

    statistic_list = [eigenvalues_SCM]
    statistic_names = [r'Eigenvalues']
    args_list = [()]

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
    results = sliding_windows_treatment_image_time_series_parallel(
        image,
        windows_mask,
        function_to_compute,
        function_args,
        multi=ENABLE_MULTI,
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns)
    print("Elpased time: %d s" %(time.time()-t_beginning))
    print('Done')

    results = results.reshape((-1,p))
    temp = np.sum(results, axis=1, keepdims=True)
    results = results / temp
    results = np.mean(results, axis=0)

    plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    plt.plot(range(1, len(results)+1), results, marker='o', linestyle='--', label=statistic_names[0], markersize=6, linewidth=1)
    
    plt.title(statistic_names[0])
    plt.show()