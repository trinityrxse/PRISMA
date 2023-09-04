#Analysis Packages
import numpy as np
import pandas as pd
from analysis import *

#Plotting
import matplotlib.pyplot as plt

#Data Formatting
from sklearn.preprocessing import RobustScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Classification Report
from sklearn.metrics import classification_report

#Load Cross Validation
from ML_Kfold import KFold_CV
#Load Model
import joblib
import torch
import numba as nb
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
import keras

#TODO get weights to work for the different samples
weight_sig, weight_bkg, lumi = get_weights() #weights & luminosity

# df of all SM data
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")
df = df[df["Type"] != 2]
df.insert(loc=0, column="SM", value=1)

data_list = ["cHq1_-1", "cHq1_1", "cHq3_-1", "cHq3_1", "cHW_-1",
             "cHW_1", "cW_-1", "cW_1"]

number_events = ["0[0-4]", "0[0-4]", "0[0-4]", "0[0-9]", "[0-1]?", "[0-1]?", "[0-1]?", "[0-1]?"]
for i in range(0, len(data_list)):
    df_EFT = pd.read_csv(f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_{data_list[i]}.csv")
    df_EFT = df_EFT[df_EFT["Type"] != 2]
    df_EFT.insert(loc=0, column="SM", value=0)

    weight_sig_EFT = get_weights_EFT(data_list[i], number_events[i])

    """""
    plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_EFT, weight2=weight_sig_EFT,
             name2=f'{data_list[i]}')
    plothist(df, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30, df2=df_EFT, weight2=weight_sig_EFT,
             name2=f'{data_list[i]}')
    plothist(df, ["m_lll"], 1500., 50, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["sumPT"], 1500., 100, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["n_btag"], 10., 9, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["met"], 500., 100, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    plothist(df, ["max_PT_jet"], 500., 100, df2=df_EFT, weight2=weight_sig_EFT, name2=f'{data_list[i]}')
    """""

    df_EFT_NN = pd.concat([df, df_EFT], axis=0, ignore_index=True)
    df_EFT_NN = df_EFT_NN.dropna(axis=0)

    x = df_EFT_NN.drop("SM", axis=1)
    x = x.drop("Type", axis=1)
    sc = RobustScaler()
    x = pd.DataFrame(sc.fit_transform(x))

    Y = df_EFT_NN['SM'].values

    X = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    filenameEFT = f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_EFT.sav"
    model_EFT = joblib.load(filenameEFT)

    y_pred = model_EFT(X)
    y_array = y.detach().numpy()
    y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using mode
    y_array = nb.typed.List([i[0] for i in y_array])
    y_pred = y_pred.detach().numpy()
    y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
    y_pred = nb.typed.List([i[0] for i in y_pred])

    plot_nodes_2(y_array, y_pred, get_weights_EFT(data_list[i], number_events[i]))

    # make df of post-training data

    featurenames = ['PT_l0', 'PT_l1', 'PT_l2', 'met',
                       'delR_l0l1', 'delR_l0l2', 'delR_l1l2',
                       'delEta_l0l1', 'delEta_l0l2', 'delEta_l1l2',
                       'dPhi_MET_l0', 'dPhi_MET_l1', 'dPhi_MET_l2',
                       'z0sintheta_l0', 'z0sintheta_l1', 'z0sintheta_l2',
                       'n_btag', 'max_PT_jet',
                       'mT_l0l1', 'mT_l0l2', 'mT_l1l2', 'sumPT',
                       'm_l0l1', 'm_l0l2', 'm_l1l2',
                       'm_lll', 'F_alpha']

    x = np.array(sc.inverse_transform(x))
    df_EFT_NN_test = pd.DataFrame(x, columns=featurenames)
    df_EFT_NN_test.insert(loc=0, column="Type", value=df_EFT_NN["Type"])
    df_EFT_NN_test.insert(loc=0, column='SM', value=y_pred_class)

    # histograms to compare input and output hists
    plothist(df_EFT_NN, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_EFT_NN_test, weight2=weight_sig_EFT,
             name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30, df2=df_EFT_NN_test, weight2=weight_sig_EFT,
             name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["m_lll"], 1500., 50, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["sumPT"], 1500., 100, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["n_btag"], 10., 9, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["met"], 500., 100, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
    plothist(df_EFT_NN, ["max_PT_jet"], 500., 100, df2=df_EFT_NN_test, weight2=weight_sig_EFT, name2=f'{data_list[i]} after')
