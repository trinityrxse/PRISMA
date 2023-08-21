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
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
import keras

#TODO get weights to work for the different samples
weight_sig, weight_bkg, lumi = get_weights() #weights & luminosity

# df of all SM data
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")
#df for the 3 categories
df_VH = df[df['Type'] == 0]
df_WWW = df[df['Type'] == 1]
df_bkg = df[df['Type'] == 2]

# load trained ML model optimised for both VH and WWW separation
filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'
model = joblib.load(filename)
model.summary()

# Use model on SM data
# format data to use model on it
x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.1)

# Get df post-training using the classifications predicted by the model 'y_test_class'
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)
y_pred = np.concatenate((y_pred_train, y_pred_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

data_list = ["cHq1_-1", "cHq1_1", "cHq3_-1", "cHq3_1", "cHW_-1",
             "cHW_1", "cW_-1", "cW_1"]

number_events = ["0[0-4]", "0[0-4]", "0[0-4]", "0[0-9]", "[0-1]?", "[0-1]?", "[0-1]?", "[0-1]?"]
for i in range(0, len(data_list)):
    df_EFT = pd.read_csv(f"/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_{data_list[i]}.csv")
    weight_sig_EFT = get_weights_EFT(data_list[i], number_events[i])

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
    # format data to use model on it
    x_EFT = df_EFT.drop('Type', axis=1)
    sc = RobustScaler()
    x_EFT = pd.DataFrame(sc.fit_transform(x_EFT))

    Y_EFT = df_EFT['Type'].values
    y_cat_EFT = to_categorical(Y_EFT, num_classes=3)

    x_train_EFT, x_test_EFT, y_train_EFT, y_test_EFT = train_test_split(x_EFT, y_cat_EFT, test_size=0.1)

    # get df post-training using the classifications predicted by the model 'y_test_class'
    y_pred_test_EFT = model.predict(x_test_EFT)
    y_pred_train_EFT = model.predict(x_train_EFT)
    y_pred_EFT = np.concatenate((y_pred_train_EFT, y_pred_test_EFT), axis=0)
    y_EFT = np.concatenate((y_train_EFT, y_test_EFT), axis=0)

    sigoverbkg('VH', y_pred_EFT, y_EFT, weight=weight_sig_EFT)
    sigoverbkg('WWW', y_pred_EFT, y_EFT, weight=weight_sig_EFT)

    sigoverbkg('VH', y_pred, y, y_pred_EFT=y_pred_EFT, y_test_EFT=y_EFT, weightEFT=weight_sig_EFT, nameEFT=data_list[i])
    sigoverbkg('WWW', y_pred, y, y_pred_EFT=y_pred_EFT, y_test_EFT=y_EFT, weightEFT=weight_sig_EFT, nameEFT=data_list[i])

    #plot_nodes(node_outputs(y_test, y_pred))
    #plot_nodes(node_outputs(y_test_EFT, y_pred_EFT))

    plot_nodes(y, y_pred, y_EFT, y_pred_EFT, weight_sig_EFT, data_list[i])

    
    # make df of post-training data
    df_test = KFold_CV(df, model)
    df_VH_test = df_test[df_test['Type'] == 0]
    df_WWW_test = df_test[df_test['Type'] == 1]
    df_bkg_test = df_test[df_test['Type'] == 2]
    
    #stacked histograms to compare input and output hists
    plothisttriple_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, ["PT_l0", "PT_l1", "PT_l2"], 400., 400., 400., 50, weight_sig, weight_bkg, lumi)
    plothisttriple_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, ["m_l0l1", "m_l0l2", "m_l0l2"], 300., 300., 300., 50, weight_sig, weight_bkg, lumi)
    plothisttriple_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 400., 400., 400., 16, weight_sig, weight_bkg, lumi)
    plothisttriple_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 4., 4., 30, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "m_lll", 0., 1500., 20, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "sumPT", 0., 1500., 100, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "n_btag", 0., 10., 9, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "met", 0., 500., 100, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "F_alpha", 0., 1., 20, weight_sig, weight_bkg, lumi)
    plothist_stacked(df_VH, df_WWW, df_bkg, df_VH_test, df_WWW_test, df_bkg_test, "z0sintheta_l0", 0., .04, 20, weight_sig, weight_bkg, lumi)
    """""