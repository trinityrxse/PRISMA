'''

Acquires all plots and prints out all significances
Runs only in top level environment
Will automatically acquire dataset csvs and models if not already on computer

'''

#Analysis Packages
import numpy as np
import pandas as pd
from selection_criteria import *
from plotting import plothist
from MLP_EFT_optimiser import *
from pre_processing_ import *
from MLP_torch_optimiser import *
from output_nodes import *
from analysis_ import *

#Plotting
import matplotlib.pyplot as plt

#Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Classification Report
from sklearn.metrics import classification_report

#Load Model
from pyhf.contrib.viz import brazil
import joblib
from scipy.stats import norm
import os

if __name__ == '__main__':

    filepath = '/localscratch' # put your filepath where you want figures / models / data saved

    best_MLP = 10 # look by eye at graphs and choose the best for node distinction
    best_EFT_C = 3 # same here
    
    if os.path.exists(f"{filepath}/df_preprocessed_.csv") is False: # check datasets exist
        pre_processing_loadout(filepath)

    if os.path.exists(f"{filepath}/MLP_torch_12.sav") is False: # check multiclassifier models exist
        best_MLP = train_MLP_multiclassifier(filepath)

    if os.path.exists(f"{filepath}/MLP_EFT_1.sav") is False: # check EFT models exist
        best_EFT_C = train_EFT_model(filepath)

    else:

        print('Best config for multiclassifier: ', best_MLP)
        print('Best config for EFT / SM distinction: ', best_EFT_C)

        #test significance of Higgs signal above bkg
        sigma_VH = multiclassifier_results(filepath, best_MLP, full_graphs=True)

        print('Reject no Higgs hypothesis with significance of ', sigma_VH, r' $\sigma$')

        data_list = ["cHq1_-1", "cHq1_1", "cHq3_-1", "cHq3_1", "cHW_-1",
                "cHW_1", "cW_-1", "cW_1"]


        # using the MLP for EFT vs SM distinction
        # calculating significance of EFT above SM 
        p_values = []
        data = []
        models = []
        twice_nll_opt = []
        sigmas = []
        for i in range(0, len(data_list)):
            p_val, datum, model, \
            twice_nll_at_best_fit, sigma = test_significance_EFT(filepath, best_MLP, best_EFT_C, data_list[i])
            p_values.append(p_val)
            data.append(datum)
            models.append(model)
            twice_nll_opt.append(twice_nll_at_best_fit)
            sigmas.append(sigma)

        for i in range(0, len(data_list)):
            print('P-Value of ', data_list[i], ': ', p_values[i])
            print('Significance of ', data_list[i], ': ', sigmas[i], r' $\sigma$')

        log_plot_multi(filepath, data, models, twice_nll_opt, data_list)



