##############################################################################
# Compute rank statistics
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

import random

if __name__ == '__main__':
    # Activate latex in figures (or not)
    LATEX_IN_FIGURES = False
    if LATEX_IN_FIGURES:
      enable_latex_infigures()

    # data

    # DEBUG mode for fast debugging (use a small patch of 200x200 pixels)
    DEBUG = False
    PATH = 'data/UAVSAR/'
    FULL_TIME_SERIES = False # if true: use the full time series, else: use only the first and last images of the time series

    image, _, X, Y = load_UAVSAR(PATH, DEBUG, FULL_TIME_SERIES)

    # Parameters
    n_r, n_c, p, T = image.shape
    windows_mask = np.ones((5,5))
    m_r, m_c = windows_mask.shape
    N = m_r*m_c

    METHODS = ['AIC', 'BIC', 'AICc', 'EDC']

    while True:
        i_r = random.choice(list(range(int(m_r/2), n_r-int(m_r/2))))
        i_c = random.choice(list(range(int(m_c/2), n_c-int(m_c/2))))

        # Obtaining data corresponding to the neighborhood defined by the mask
        local_data = image[i_r-int(m_r/2):i_r+int(m_r/2)+1, i_c-int(m_c/2):i_c+int(m_c/2)+1, :, :]
        local_data = np.transpose(local_data, [2, 0, 1, 3])
        local_data = local_data.reshape((p,N*T))

        # Computing the function over the local data
        for method in METHODS:
          criterion = SCM_rank_criterion(local_data, method=method)
          ranks = np.arange(criterion.shape[0])+1
          plt.plot(ranks, criterion, label=method)
          idx_min = criterion.argmin()
          plt.plot(ranks[idx_min],criterion[idx_min],'o')

        plt.legend()
        plt.show()