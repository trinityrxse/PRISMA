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
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
import keras

# df of all data provided
# can read in dfs from EFTs
df = pd.read_csv("/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/df_preprocessed_.csv")

"""""
#plot histograms for all data
plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50)
plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50)
plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30)
plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30)
plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30)
plothist(df, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30)
plothist(df, ["m_lll"], 1500., 50)
plothist(df, ["sumPT"], 1500., 100)
plothist(df, ["n_btag"], 10., 9)
plothist(df, ["met"], 500., 100)
plothist(df, ["max_PT_jet"], 500., 100)
plothist(df, ["F_alpha"], 1., 20)
"""""

# load trained ML model optimised for both VH and WWW separation
filename = "/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/MLP_torch_new.sav"
#filename = '/Users/trinitystenhouse/Documents/University_MSci/2022-3/PRISMA_code/initial_model.sav'
model = joblib.load(filename)

# format data to use model on it
x = df.drop('Type', axis=1)
sc = RobustScaler()
x = pd.DataFrame(sc.fit_transform(x))

Y = df['Type'].values
y_cat = to_categorical(Y, num_classes=3)

x_train, x_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.2)

X = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y_cat, dtype=torch.float32).reshape(-1, 3)
y_int = torch.tensor(y_cat, dtype=torch.int32).reshape(-1, 3)

"""""
#model evaluation
pred_train = model.predict(x_train)
scores = model.evaluate(x_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
pred_test = model.predict(x_test)
scores2 = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))
"""""

# Get df post-training using the classifications predicted by the model 'y_test_class'
y_pred = model(X)
y_array = y.detach().numpy()
y_pred = y_pred.detach().numpy()

VH_sig_over_bkg = sigoverbkg('VH', y_pred, y_array)
WWW_sig_over_bkg = sigoverbkg('WWW', y_pred, y_array)

plot_nodes(y_array, y_pred)

y_pred_class = np.argmax(y_pred, axis=1)  # classes predicted for training data
y_truth_class = np.argmax(y_array, axis=1)  # classes predicted for test data using model


print(classification_report(y_truth_class, y_pred_class))

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
df_test = pd.DataFrame(x, columns=featurenames)
df_test.insert(loc=0, column='Type', value=y_pred_class)

# TODO fix K fold CV for pytorch network
# make df of post-training data
#df_test = KFold_CV(df, model)

weight_sig = get_weights()[0]
#histograms to compare input and output hists
plothist(df, ["PT_l0", "PT_l1", "PT_l2"], 400., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_l0l1", "m_l0l2", "m_l1l2"], 300., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["mT_l0l1", "mT_l0l2", "mT_l1l2"], 1200, 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delR_l0l1", "delR_l0l2", "delR_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["delEta_l0l1", "delEta_l0l2", "delEta_l1l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["z0sintheta_l0", "z0sintheta_l1", "z0sintheta_l2"], 4., 30, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["m_lll"], 1500., 50, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["sumPT"], 1500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["n_btag"], 10., 9, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["met"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["max_PT_jet"], 500., 100, df2=df_test, weight2=weight_sig, name2='Post-ML')
plothist(df, ["F_alpha"], 1., 20, df2=df_test, weight2=weight_sig, name2='Post-ML')

