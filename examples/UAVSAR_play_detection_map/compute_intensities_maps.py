##############################################################################
# Compute and save detection map as numpy array.
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
from datetime import datetime

# The code is already multi threaded so we block OpenBLAS multi thread.
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import path of root repo
temp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, temp)
temp = os.path.join(temp, 'examples', 'UAVSAR')
sys.path.insert(1, temp)

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
import tikzplotlib

from LRST.generic_functions import *
from LRST.monte_carlo_tools import *
from LRST.multivariate_images_tools import *
from LRST.change_detection_functions import *
from LRST.low_rank_statistics import *
from LRST.proportionality_statistics import *
from LRST.read_sar_data import *
from LRST.wavelet_functions import *
from compute_ROC_UAVSAR_dataset import download_uavsar_cd_dataset, load_UAVSAR 

if __name__ == '__main__':
    # Activate latex in figures (or not)
    LATEX_IN_FIGURES = True
    if LATEX_IN_FIGURES:
        enable_latex_infigures()

    USE_TIKZPLOTLIB = True

    # date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # DIRECTORY_TIKZ = 'tikz_figures' + '_' + date
    DIRECTORY_TIKZ = 'tex'
    if USE_TIKZPLOTLIB:
        if not os.path.exists(DIRECTORY_TIKZ):
            os.makedirs(DIRECTORY_TIKZ)

    # Enable parallel processing (or not)
    ENABLE_MULTI = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 2
    number_of_threads_columns = 4
    if os.cpu_count()!=(number_of_threads_rows*number_of_threads_columns):
        print('WARNING: not all cpu are used')
    
    # data

    # DEBUG mode for fast debugging (use a small patch)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    FULL_TIME_SERIES = False # if true: use the full time series, else: use only the first and last images of the time series
    SCENE_NUMBER = 2

    image, _, _, _ = load_UAVSAR(PATH, DEBUG, FULL_TIME_SERIES, SCENE_NUMBER)
    
    # Parameters
    n_r, n_rc, p, T = image.shape
    windows_mask = np.ones((7,7))
    m_r, m_c = windows_mask.shape
    function_to_compute = compute_several_statistics

    # Comparaison of the 4 models
    statistic_list = [covariance_equality_glrt_gaussian_statistic, LR_CM_equality_test, scale_and_shape_equality_robust_statistic, scale_and_shape_equality_robust_statistic_low_rank]
    statistic_names = ['Gaussian', 'Low_Rank_Gaussian', 'Compound', 'Low_Rank_Compound']
    args_list = ['log', (3, False, 'log'), (0.01, 100, 'log'), (0.01, 100, 3, False, 'log')]

    if not (len(statistic_list)==len(statistic_names)==len(args_list)):
        print('ERROR')
        sys.exit(1)

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
    print("Elapsed time: %d s" %(time.time()-t_beginning))
    print('Done')
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(base_path, 'intensities_maps')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = os.path.join(base_path, 'scene_'+str(SCENE_NUMBER))
    np.save(path, results)

