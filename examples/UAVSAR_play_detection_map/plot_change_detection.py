##############################################################################
# ROC plots on real data UAVSAR
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

    USE_TIKZPLOTLIB = False

    # date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    # DIRECTORY_TIKZ = 'tikz_figures' + '_' + date
    DIRECTORY_TIKZ = 'tex'
    if USE_TIKZPLOTLIB:
        if not os.path.exists(DIRECTORY_TIKZ):
            os.makedirs(DIRECTORY_TIKZ)

    # DEBUG mode for fast debugging (use a small patch)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    SCENE_NUMBER = 2
    PFA = 0.1
    if not(PFA>=0 and PFA<=1):
        print('ERROR in selected pfa')
        sys.exit(1)

    # Statistics of detection maps
    STATISTIC_NAMES = ['Gaussian', 'Low rank Gaussian', 'Compound', 'Low rank compound']

    # Path of detection map
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(base_path, 'intensities_maps')
    path_intensities = os.path.join(base_path, 'scene_'+str(SCENE_NUMBER)+'.npy')
    if not os.path.exists(path_intensities):
        print('ERROR with path of detection map')
        print('path=', path_intensities)
        sys.exit(1)
    
    # Load detection map
    intensities = np.load(path_intensities)
    
    # Load ground truth
    _, ground_truth_original, X, Y = load_UAVSAR(PATH, DEBUG, False, scene_number=SCENE_NUMBER, verbose=False)
    m_r = ground_truth_original.shape[0]-intensities.shape[0]
    m_c = ground_truth_original.shape[1]-intensities.shape[1]
    ground_truth = ground_truth_original[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]

    # Compute threshold
    NUMBER_OF_POINTS = 500
    thresholds = np.zeros(len(STATISTIC_NAMES))
    pfas = np.zeros(len(STATISTIC_NAMES))
    pds = np.zeros(len(STATISTIC_NAMES))
    detection_maps = np.zeros(intensities.shape)
    for i_s, statistic in enumerate(STATISTIC_NAMES):
        
        # Sorting values of statistic
        λ_vec = np.sort(vec(intensities[:,:,i_s]), axis=0)
        λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]

        # Selectionning NUMBER_OF_POINTS values from beginning to end
        indices_λ = np.floor(np.linspace(0, len(λ_vec)-1, num=NUMBER_OF_POINTS)) # logspace
        λ_vec = np.flip(λ_vec, axis=0)
        λ_vec = λ_vec[indices_λ.astype(int)]

        actual_pfa = 0
        i = 0
        while (actual_pfa<PFA) and (i<len(λ_vec)):
            detection_map = intensities[:,:,i_s] >= λ_vec[i]
            false_alarms = detection_map * np.logical_not(ground_truth)
            actual_pfa = false_alarms.sum() / (ground_truth==0).sum()
            true_alarms = detection_map * ground_truth
            actual_pd = true_alarms.sum() / (ground_truth==1).sum()
            i = i + 1
        pfas[i_s] = actual_pfa
        pds[i_s] = actual_pd
        print(statistic)
        print('threshold:', round(λ_vec[i], 4))
        print('pfa found:', round(actual_pfa, 4))
        print('pd found:', round(actual_pd, 4))
        print()
        # detection_maps[:, :, i_s] = detection_map 
        detection_maps[:, :, i_s] = 0.6*false_alarms + 0.8*true_alarms

    # Showing ground truth
    plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    plt.pcolormesh(X, Y, ground_truth_original, cmap='tab20c_r')
    plt.title('Ground Truth')
    plt.xlabel(r'Azimuth (m)')
    plt.ylabel(r'Range (m)')
    plt.colorbar()
    if USE_TIKZPLOTLIB:
        tikzplotlib.save(DIRECTORY_TIKZ + '/ground_truth.tex')

    # Showing statistics intensities raw
    for i_s, statistic in enumerate(STATISTIC_NAMES):
        plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
        plt.pcolormesh(X[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)], Y[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)], detection_maps[:,:,i_s], cmap='tab20c_r')
        plt.title(statistic)
        plt.xlabel(r'Azimuth (m)')
        plt.ylabel(r'Range (m)')
        plt.colorbar()
        if USE_TIKZPLOTLIB:
            tikzplotlib.save(DIRECTORY_TIKZ + '/statistic' + str(i_s) + '.tex')

    
    if not USE_TIKZPLOTLIB:
        plt.show()
