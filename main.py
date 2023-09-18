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

if __name__ == '__main__':

    filepath = '/localscratch'

    if pd.read_csv(f"{filepath}/df_preprocessed_.csv") is None:
        pre_processing_loadout(filepath)

    if joblib.load(f"{filepath}/MLP_torch_12.sav") is None:
        train_MLP_multiclassifier(filepath)

    if joblib.load(f"{filepath}/MLP_EFT_1.sav") is None:
        train_EFT_model(filepath)

    else:

        multiclassifier_results(filepath, full_graphs=True)

        data_list = ["cHq1_-1", "cHq1_1", "cHq3_-1", "cHq3_1", "cHW_-1",
                "cHW_1", "cW_-1", "cW_1"]


        # using the MLP for EFT vs SM distinction
        # calculating significance of EFT above SM 
        p_values = []
        data = []
        models = []
        twice_nll_opt = []
        for i in range(0, len(data_list)):
            p_val, datum, model, \
            twice_nll_at_best_fit = test_significance_EFT(filepath, data_list[i])
            p_values.append(p_val)
            data.append(datum)
            models.append(model)
            twice_nll_opt.append(twice_nll_at_best_fit)

        for i in range(0, len(data_list)):
            print('P-Value of ', data_list[i], ': ', p_values[i])

        log_plot_multi(filepath, data, models, twice_nll_opt, data_list)




