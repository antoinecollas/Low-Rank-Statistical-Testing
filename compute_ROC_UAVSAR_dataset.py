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
from generic_functions import *
import matplotlib.pyplot as plt
from monte_carlo_tools import *
from multivariate_images_tools import *
from change_detection_functions import *
from low_rank_statistics import *
from proportionality_statistics import *
from read_sar_data import *
from wavelet_functions import *
import os
import time
import seaborn as sns
sns.set_style("darkgrid")


def download_uavsar_cd_dataset(path='./Data/'):

    # if directory exists just catch error
    try:
        os.mkdir(path)
    except:
        path

    # Links to UAVSAR datasets
    list_of_files = [
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_09014_007_090423_L090VV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_12021_007_120427_L090VV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_14006_007_140117_L090VV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HH_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090VV_03_BC_s4_1x1.slc',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HH_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090HV_03_BC.ann',
                    'http://downloaduav2.jpl.nasa.gov/Release25/SanAnd_26524_03/SanAnd_26524_15059_006_150511_L090VV_03_BC.ann']

    for file_url in list_of_files:
        if not os.path.exists(path + file_url.split('/')[-1]):
            if not os.path.exists(path):
                os.makedirs(path)
            import wget
            print("File %s not found, downloading it" % file_url.split('/')[-1])
            wget.download(url=file_url, out=path+file_url.split('/')[-1])



if __name__ == '__main__':

    # DEBUG mode for fast debugging (use a small patch of 200x200 pixels)
    DEBUG = False

    # Activate latex in figures (or not)
    latex_in_figures = True
    if latex_in_figures:
      enable_latex_infigures()

    # Enable parallel processing (or not)
    enable_multi = True
    # These two variables serves to split the original image into sub-images to be treated in parallel
    # In general the optimal parameters are obtained for 
    # number_of_threads_rows*number_of_threads_columns = number of cores on the machine
    number_of_threads_rows = 6
    number_of_threads_columns = 8
    if os.cpu_count()!=(number_of_threads_rows*number_of_threads_columns):
        print('WARNING: not all cpu are used')

    # data
    PATH = 'Data/UAVSAR/'
    TIME_SERIES = True # if true: use the full time series, else: use only the first and last images of the time series
    # Downloading data if needed
    download_uavsar_cd_dataset(path=PATH)
    
    # Reading data using the class
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   READING     |') 
    print( '|   dataset     |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    data_class = uavsar_slc_stack_1x1(PATH)
    data_class.read_data(time_series=TIME_SERIES, polarisation=['HH', 'HV', 'VV'], segment=4, crop_indexes=[28891,31251,2891,3491])
    if DEBUG:
        n_r, n_rc, _, _ = data_class.data.shape
        new_size_image = 200
        data_class.data = data_class.data[(n_r//2)-(new_size_image//2):(n_r//2)+(new_size_image//2), (n_rc//2)-(new_size_image//2):(n_rc//2)+(new_size_image//2), :, :]
    print('Done')

    # Spatial vectors
    center_frequency = 1.26e+9 # GHz, for L Band
    bandwith = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['Bandwidth']) * 10**6 # Hz
    range_resolution = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['1x1 SLC Range Pixel Spacing']) # m, for 1x1 slc data
    azimuth_resolution = float(data_class.meta_data['SanAnd_26524_09014_007_090423_L090HH_03_BC']['1x1 SLC Azimuth Pixel Spacing']) # m, for 1x1 slc data
    number_pixels_azimuth, number_pixels_range, p, T = data_class.data.shape
    range_vec = np.linspace(-0.5,0.5,number_pixels_range) * range_resolution * number_pixels_range
    azimuth_vec = np.linspace(-0.5,0.5,number_pixels_azimuth) * azimuth_resolution * number_pixels_azimuth
    Y, X = np.meshgrid(range_vec,azimuth_vec)

    # Decomposition parameters (j=1 always because reasons)
    R = 2
    L = 2
    d_1 = 10
    d_2 = 10

    # Wavelet decomposition of the time series
    print( '|￣￣￣￣￣￣￣￣|')
    print( '|   Wavelet     |') 
    print( '| decomposition |')
    print( '|               |' )  
    print( '| ＿＿＿_＿＿＿＿|') 
    print( ' (\__/) ||') 
    print( ' (•ㅅ•) || ')
    print( ' / 　 づ')
    image = np.zeros((number_pixels_azimuth, number_pixels_range, p*R*L, T), dtype=complex)
    for t in range(T):
        for i_p in range(p):
            image_temp = decompose_image_wavelet(data_class.data[:,:,i_p,t], bandwith, range_resolution, azimuth_resolution, center_frequency,
                                    R, L, d_1, d_2)
            image[:,:,i_p*R*L:(i_p+1)*R*L, t] = image_temp
    image_temp = None
    print('Done')


    # Parameters
    n_r, n_rc, p, T = image.shape
    windows_mask = np.ones((5,5))
    m_r, m_c = windows_mask.shape
    function_to_compute = compute_several_statistics

    # Gaussian
    # statistic_list = [covariance_equality_glrt_gaussian_statistic]
    # statistic_names = [r'$\hat{\Lambda}_{\mathrm{G}}$']
    # args_list = ['log']

    # Low rank Gaussian
    # statistic_list = [LR_CM_equality_test]
    # statistic_names = [r'$\hat{\Lambda}_{\mathrm{LRG}}$']
    # args_list = [(3, None, 'log')]

    # Compound Gaussian
    # statistic_list = [scale_and_shape_equality_robust_statistic]
    # statistic_names = [r'$\hat{\Lambda}_{\mathrm{CG}}$']
    # args_list = [(0.01, 20, 'log')]

    # Low rank Compound Gaussian
    # statistic_list = [scale_and_shape_equality_robust_statistic_low_rank]
    # statistic_names = [r'$\hat{=\Lambda}_{\mathrm{LRCG}}$']
    # args_list = [(0.01, 20, 3, False, 'log')]

    # Comparaison of the 4 models
    statistic_list = [covariance_equality_glrt_gaussian_statistic, LR_CM_equality_test, scale_and_shape_equality_robust_statistic, scale_and_shape_equality_robust_statistic_low_rank]
    statistic_names = [r'$\hat{\Lambda}_{\mathrm{G}}$', r'$\hat{\Lambda}_{\mathrm{LRG}}$', r'$\hat{\Lambda}_{\mathrm{CG}}$', r'$\hat{\Lambda}_{\mathrm{LRCG}}$']
    args_list = ['log', (3, False, 'log'), (0.01, 20, 'log'), (0.01, 20, 3, False, 'log')]

    # Comparaison of 2 methods of evaluating σ2
    # statistic_list = [scale_and_shape_equality_robust_statistic_low_rank, scale_and_shape_equality_robust_statistic_low_rank]
    # statistic_names = [r'$\hat{\Lambda}_{\mathrm{LRCG}, \sigma2 \mathrm{a priori}}$', r'$\hat{\Lambda}_{\mathrm{LRCG}, \sigma2 \mathrm{GLRT}}$']
    # args_list = [(0.01, 20, 3, True, 'log'), (0.01, 20, 3, False, 'log')]

    # Test the robustness to the rank of LRCG
    # rank_list = [(i+1) for i in range(p)]
    # statistic_list = [scale_and_shape_equality_robust_statistic_low_rank for i in range(len(rank_list))]
    # statistic_names = ['$\hat{\Lambda}_{\mathrm{LRCG, R='+str(rank)+'}}$' for rank in rank_list]
    # args_list = [(0.01, 20, rank, False, 'log') for rank in rank_list]

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
        multi=enable_multi,
        number_of_threads_rows=number_of_threads_rows,
        number_of_threads_columns=number_of_threads_columns)
    print("Elpased time: %d s" %(time.time()-t_beginning))
    print('Done')


    # Computing ROC curves
    number_of_points = 30
    ground_truth_original = np.load('./Data/ground_truth_uavsar_scene1.npy')
    if DEBUG:
        ground_truth_original = ground_truth_original[(n_r//2)-(new_size_image//2):(n_r//2)+(new_size_image//2), (n_rc//2)-(new_size_image//2):(n_rc//2)+(new_size_image//2)]
    ground_truth = ground_truth_original[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)]
    pfa_array = np.zeros((number_of_points, len(function_args[0])))
    pd_array = np.zeros((number_of_points, len(function_args[0])))
    for i_s, statistic in enumerate(statistic_names):

        # Sorting values of statistic
        λ_vec = np.sort(vec(results[:,:,i_s]), axis=0)
        λ_vec = λ_vec[np.logical_not(np.isinf(λ_vec))]

        # Selectionning number_of_points values from beginning to end
        indices_λ = np.floor(np.linspace(0, len(λ_vec)-1, num=number_of_points)) # logspace
        λ_vec = np.flip(λ_vec, axis=0)
        λ_vec = λ_vec[indices_λ.astype(int)]

        # Thresholding and summing for each value
        for i_λ, λ in enumerate(λ_vec):
            good_detection = (results[:,:,i_s] >= λ) * ground_truth
            false_alarms = (results[:,:,i_s] >= λ) * np.logical_not(ground_truth)
            pd_array[i_λ, i_s] = good_detection.sum() / (ground_truth==1).sum()
            pfa_array[i_λ, i_s] = false_alarms.sum() / (ground_truth==0).sum()

    # Showing images 
    Amax = 20*np.log10((np.sum(np.abs(image[:,:,:,:])**2, axis=2)/p).max())
    Amin = 20*np.log10((np.sum(np.abs(image[:,:,:,:])**2, axis=2)/p).min())
    for t in range(T):
        Span = np.sum(np.abs(image[:,:,:,t])**2, axis=2)/p
        plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
        plt.pcolormesh(X,Y,20*np.log10(Span), cmap='bone', vmin=Amin, vmax=Amax)
        plt.title(r'Image at $t_%d$' %(t+1))
        plt.xlabel(r'Azimuth (m)')
        plt.ylabel(r'Range (m)')
        plt.colorbar()

    # Showing ground truth
    plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    plt.pcolormesh(X,Y,ground_truth_original, cmap='tab20c_r')
    plt.title('Ground Truth')
    plt.xlabel(r'Azimuth (m)')
    plt.ylabel(r'Range (m)')
    plt.colorbar()

    # Showing statistics results raw
    for i_s, statistic in enumerate(statistic_names):
        plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
        image_temp = np.nan*np.ones((number_pixels_azimuth, number_pixels_range))
        image_temp[int(m_r/2):-int(m_r/2), int(m_c/2):-int(m_c/2)] = (results[:,:,i_s] - results[:,:,i_s].min())# / (results[:,:,i_s].max() - results[:,:,i_s].min())
        plt.pcolormesh(X,Y, image_temp, cmap='jet')
        plt.xlabel(r'Azimuth (m)')
        plt.ylabel(r'Range (m)')
        # plt.title(statistic)
        plt.colorbar()

    # Showing statistics results ROC
    markers = ['o', 's', 'h', '*', 'd', 'p']
    plt.figure(figsize=(6, 4), dpi=120, facecolor='w')
    for i_s, statistic in enumerate(statistic_names):
        plt.plot(pfa_array[:,i_s], pd_array[:,i_s], linestyle='--', label=statistic, markersize=4, linewidth=1)
    plt.xlabel(r'$\mathrm{P}_{\mathrm{FA}}$')
    plt.ylabel(r'$\mathrm{P}_{\mathrm{D}}$')
    plt.legend()
    plt.xlim([0.,1])
    plt.ylim([0,1])
    plt.show()